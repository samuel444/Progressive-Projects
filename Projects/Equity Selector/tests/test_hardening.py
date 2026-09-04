import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from equity_selector.database import quote_identifier, read_table, write_frame
from equity_selector.metrics import performance_metrics, relative_metrics
from equity_selector.parameters import parse_parameters, parameters_to_json, unique_models
from equity_selector.portfolio import capped_weights, portfolio_returns_from_scores
from equity_selector.validation import purge_training_data, train_validation_test_split
from features.beta import beta_ratios
from screening.target_feature_screening import quantile_spread
from targets.volatility import future_upside_downside_volatility


def panel():
    return pd.DataFrame(
        {
            "Date": np.repeat(pd.date_range("2024-01-01", periods=3), 2),
            "Ticker": ["A", "B"] * 3,
            "Return": [0.9, -0.8, -0.2, 0.4, 0.3, -0.1],
            "Stock_Score": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        }
    )


def test_scores_earn_next_period_returns_and_do_not_mutate():
    data = panel().sample(frac=1, random_state=7)
    original = data.copy(deep=True)
    result = portfolio_returns_from_scores(data, max_weight=1)
    assert result["Return"].tolist() == pytest.approx([-0.2, -0.1])
    assert result["A"].tolist() == [1, 0]
    assert result["B"].tolist() == [0, 1]
    assert result["Date"].tolist() == list(pd.date_range("2024-01-02", periods=2))
    assert_frame_equal(data, original)


def test_caps_cash_fees_and_first_missing_return():
    data = panel()
    data.loc[:1, "Return"] = np.nan
    result = portfolio_returns_from_scores(data, max_weight=0.3, trading_fee=0.01)
    assert result["Return"].tolist() == pytest.approx([-0.06 - 0.003, -0.03 - 0.006])
    assert result[["A", "B"]].sum(axis=1).tolist() == pytest.approx([0.3, 0.3])


def test_missing_held_return_policy_and_duplicate_rows():
    data = panel()
    data.loc[2, "Return"] = np.nan
    assert portfolio_returns_from_scores(data, max_weight=1)["Return"].iloc[0] == 0
    with pytest.raises(ValueError, match="Missing held return"):
        portfolio_returns_from_scores(data, max_weight=1, missing_return="raise")
    with pytest.raises(ValueError, match="Duplicate"):
        portfolio_returns_from_scores(pd.concat([data, data.iloc[:1]]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_weight": 0},
        {"max_weight": np.nan},
        {"concentration_penalty": 1.1},
        {"trading_fee": -1},
    ],
)
def test_invalid_portfolio_settings(kwargs):
    with pytest.raises(ValueError):
        portfolio_returns_from_scores(panel(), **kwargs)


def test_no_strategies_and_one_date_have_stable_outputs():
    data = panel()
    data["Stock_Score"] = 0
    assert portfolio_returns_from_scores(data)["Return"].tolist() == [0.0, 0.0]
    assert list(portfolio_returns_from_scores(data.iloc[:0]).columns) == ["Date", "Return"]
    assert portfolio_returns_from_scores(data.iloc[:2]).empty


def test_capped_weights_expected_distribution():
    weights = capped_weights(pd.Series([8.0, 1.0, 1.0], index=["A", "B", "C"]), 0.5, 0)
    assert weights.to_dict() == pytest.approx({"A": 0.5, "B": 0.25, "C": 0.25})
    assert capped_weights(pd.Series([1e308, 1e308]), 1).tolist() == [0.5, 0.5]


def test_metrics_include_initial_capital_and_periodic_sharpe():
    returns = pd.Series([-0.2, 0.25, 0.1])
    metrics = performance_metrics(returns)
    assert metrics["Return"] == pytest.approx(0.1)
    assert metrics["Max Drawdown"] == pytest.approx(-0.2)
    assert metrics["Average Drawdown"] == pytest.approx(-0.2 / 3)
    assert metrics["Sharpe Ratio"] == pytest.approx(
        np.mean([-0.2, 0.25, 0.1]) / np.std([-0.2, 0.25, 0.1], ddof=1) * np.sqrt(252)
    )
    assert np.isnan(performance_metrics([0.0, 0.0])["Sharpe Ratio"])
    assert performance_metrics([])["Return"] == 0
    assert performance_metrics([-1.0, 0.0])["Max Drawdown"] == -1


@pytest.mark.parametrize(
    "value",
    [
        {"class_weight": "null", "name": "nan", "flag": True, "hidden_layer_sizes": (20, 10)},
        '{"class_weight":"null","name":"nan","flag":true,"hidden_layer_sizes":[20,10]}',
        "{'class_weight': 'null', 'name': 'nan', 'flag': True, 'hidden_layer_sizes': (20, 10)}",
    ],
)
def test_parameter_roundtrip_preserves_strings_and_tuples(value):
    expected = {"class_weight": "null", "name": "nan", "flag": True, "hidden_layer_sizes": (20, 10)}
    assert parse_parameters(value) == expected
    assert parse_parameters(parameters_to_json(value)) == expected


def test_numpy_legacy_parameters_and_reject_code():
    assert parse_parameters("{'alpha': np.float64(0.1), 'depth': nan}") == {
        "alpha": 0.1,
        "depth": None,
    }
    for value in ["__import__('os').getcwd()", "[1, 2]", "42", '{"x": some_call()}']:
        with pytest.raises(ValueError):
            parse_parameters(value)
    models = [
        {"name": "MLP", "params": {"hidden_layer_sizes": (10,)}},
        {"name": "MLP", "params": '{"hidden_layer_sizes": [10]}'},
    ]
    copied = unique_models(models)
    assert len(copied) == 1
    copied[0]["params"]["hidden_layer_sizes"] = (99,)
    assert models[0]["params"]["hidden_layer_sizes"] == (10,)


def test_database_roundtrip_quotes_rollback_and_schema_guard(tmp_path):
    path = tmp_path / "fixture.db"
    original = pd.DataFrame(
        {"Parameters": [parameters_to_json({"hidden_layer_sizes": (5, 2)})], "Value": [1.0]}
    )
    with sqlite3.connect(path) as connection:
        write_frame(original, 'quoted " table', connection, if_exists="replace")
    restored = read_table(path, 'quoted " table')
    assert_frame_equal(original, restored)
    assert parse_parameters(restored.Parameters.iloc[0])["hidden_layer_sizes"] == (5, 2)
    with pytest.raises(RuntimeError):
        with sqlite3.connect(path) as connection:
            write_frame(
                pd.DataFrame({"Changed": [2]}), 'quoted " table', connection, if_exists="replace"
            )
            write_frame(pd.DataFrame({"Other": [3]}), "second", connection, if_exists="replace")
            raise RuntimeError("Injected failure after both writes")
    assert_frame_equal(read_table(path, 'quoted " table'), original)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='second'").fetchone()
            is None
        )
        connection.execute(
            "CREATE INDEX custom_index ON " + quote_identifier('quoted " table') + "(Value)"
        )
        with pytest.raises(ValueError, match="custom schema"):
            write_frame(original, 'quoted " table', connection, if_exists="replace")
        with pytest.raises(ValueError, match="schema mismatch"):
            write_frame(
                pd.DataFrame({"Wrong": [1]}), 'quoted " table', connection, if_exists="append"
            )
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing.db", "anything")
    assert not (tmp_path / "missing.db").exists()


def test_chronological_split_and_zero_or_excess_purge():
    data = pd.DataFrame(
        {"Date": np.repeat(pd.date_range("2020-01-01", periods=10), 2), "Ticker": ["A", "B"] * 10}
    )
    train, valid, test = train_validation_test_split(data.sample(frac=1, random_state=3))
    assert (len(train), len(valid), len(test)) == (12, 4, 4)
    assert train.Date.max() < valid.Date.min() < test.Date.min()
    assert_frame_equal(purge_training_data(train, 0), train)
    assert purge_training_data(train, 2).Date.max() == pd.Timestamp("2020-01-04")
    assert purge_training_data(train, 6).empty
    for kwargs in [{"test": 0}, {"validation": 0.9}, {"test": np.nan}]:
        with pytest.raises(ValueError):
            train_validation_test_split(data, **kwargs)


def test_shortened_future_volatility_labels_are_missing():
    data = pd.DataFrame({"Close": [100.0, 102.0, 101.0, 104.0, 103.0, 107.0, 105.0, 110.0, 106.0]})
    result = future_upside_downside_volatility(data, horizons=5)
    assert result.filter(like="Future").tail(5).isna().all().all()
    expected = pd.Series([0.02, 104 / 101 - 1, 107 / 103 - 1]).std()
    assert result["Future Upside Volatility 5"].iloc[0] == pytest.approx(expected)


def test_screening_empty_and_constant_features_disqualifies():
    data = pd.DataFrame({"constant": [1, 1, 1], "missing": [np.nan] * 3, "target": [1, 2, 3]})
    assert quantile_spread(data, ["constant", "missing"], "target") == ([], ["constant", "missing"])


def test_beta_ratio_accepts_integer_with_default_tuple():
    data = pd.DataFrame(
        {"Close": np.linspace(100, 130, 300)}, index=pd.bdate_range("2023-01-01", periods=300)
    )
    result = beta_ratios(data, short_windows=20, market_df=data.copy())
    assert result["Beta Ratio 20 120"].dropna().iloc[-1] == pytest.approx(1)


def test_precise_uses_shared_portfolio_and_metrics():
    spec = importlib.util.spec_from_file_location(
        "precise_fixture", Path(__file__).parents[1] / "equity_selector/stages/precise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.portfolio_returns_from_scores is portfolio_returns_from_scores
    module.market_return = 0.1
    module.market_sharpe = 1
    module.market_max_drawdown = -0.1
    module.market_average_drawdown = -0.05
    data = pd.DataFrame({"Return": [-0.2, 0.25, 0.1]})
    before = data.copy()
    actual = module.result_metrics(data)
    assert actual[0] == pytest.approx(0.1)
    assert actual[3] == pytest.approx(-0.2)
    assert_frame_equal(data, before)

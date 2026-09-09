import sqlite3
import ast
import pandas as pd
from equity_selector.macro_regimes import build_macro_regimes, confirm_regimes, save_macro_regimes


def test_confirmation_no_backdate_and_missing_hold():
    assert confirm_regimes(["Expansion", "Contraction", "Contraction", "Contraction"], 3) == [
        "Expansion"
    ] * 3 + ["Contraction"]
    assert confirm_regimes(
        ["Unknown", "Expansion", "Unknown", "Contraction", "Unknown", "Contraction", "Contraction"],
        2,
    ) == ["Unknown", "Expansion", "Expansion", "Expansion", "Expansion", "Expansion", "Contraction"]


def test_sparse_signals_and_prefix_invariance():
    f = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=10),
            "Macro PIT Retail_Sales Growth 3M Percent": [1] * 4 + [-1] * 6,
            "Macro PIT Industrial_Production Growth 3M Percent": [1] * 4 + [-1] * 6,
            "Macro PIT M2 Growth 6M Percent": 1,
        }
    )
    _, daily = build_macro_regimes(f, 3, return_daily=True)
    assert daily.Growth.tolist() == ["Expansion"] * 6 + ["Contraction"] * 4
    assert daily.Liquidity.eq("Expanding").all()
    assert daily.Housing.eq("Unknown").all()
    for n in range(1, 11):
        _, prefix = build_macro_regimes(f.iloc[:n], 3, return_daily=True)
        pd.testing.assert_frame_equal(prefix, daily.iloc[:n])


def test_database_discovery_deterministic_literal(tmp_path):
    db = tmp_path / "macro_data.db"
    f = pd.DataFrame(
        {"Date": ["2020-01-01", "2020-01-02"], "Macro PIT M2 Growth 6M Percent": [1, -1]}
    )
    with sqlite3.connect(db) as c:
        f.to_sql("existing macro table", c, index=False)
        pd.DataFrame({"x": [1]}).to_sql("unrelated", c, index=False)
    expected = save_macro_regimes(db, output_dir=tmp_path)
    first = (tmp_path / "Macro_Regimes.txt").read_bytes()
    save_macro_regimes(db, output_dir=tmp_path)
    assert first == (tmp_path / "Macro_Regimes.txt").read_bytes()
    assert ast.literal_eval(first.decode()) == expected


def test_macro_alignment_never_uses_future_row():
    from equity_selector.macro_regimes import align_macro_features

    stocks = pd.DataFrame(
        {"Date": ["2020-01-04", "2020-01-01", "2020-01-03"], "Ticker": ["A", "A", "B"]}
    )
    macro = pd.DataFrame(
        {"Date": ["2020-01-02", "2020-01-05"], "Macro PIT M2 Growth 6M Percent": [1, 100]}
    )
    aligned = align_macro_features(stocks, macro)
    assert aligned.Date.tolist() == stocks.Date.tolist()
    assert aligned.iloc[0, 2] == 1 and pd.isna(aligned.iloc[1, 2]) and aligned.iloc[2, 2] == 1

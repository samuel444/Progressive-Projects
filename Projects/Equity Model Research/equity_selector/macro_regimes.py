"""Causal, missing-signal-tolerant classification of existing Macro PIT features.

Intervals are inclusive ISO dates. Confirmation counts observations, not elapsed
calendar days. Initial valid classification is active immediately; subsequent
changes need confirmation. Unknown observations hold the active state and reset
pending candidates. No interpolation, backfill or revised-level fallback occurs.
"""

import ast
from pathlib import Path
from pprint import pformat
import sqlite3
import numpy as np
import pandas as pd
from .config import data_root

PREFIX = "Macro PIT "
FAMILIES = {
    "Growth": (
        "Expansion",
        "Neutral",
        "Contraction",
        2,
        [
            ("Industrial_Production Growth 3M Percent", 1, 0.1),
            ("Retail_Sales Growth 3M Percent", 1, 0.1),
            ("Payrolls Growth 3M Percent", 1, 0.1),
            ("Real_GDP Growth 1Q Annualized Percent", 1, 0.1),
            ("Consumer_Sentiment Growth 3M Versus Prior 3M PP", 1, 0.1),
            ("Housing_Starts Growth 3M Versus Prior 3M PP", 1, 0.1),
        ],
    ),
    "Inflation": (
        "Rising",
        "Stable",
        "Falling",
        2,
        [(name + " YoY Acceleration 3M PP", 1, 0.05) for name in ["CPI", "Core_CPI", "PCE"]],
    ),
    "Labour": (
        "Strong",
        "Neutral",
        "Deteriorating",
        2,
        [
            ("Payrolls Growth 3M Percent", 1, 0.1),
            ("Payrolls YoY Acceleration 3M PP", 1, 0.05),
            ("Unemployment Change 1M PP", -1, 0.05),
            ("Unemployment Three Month Mean Minus Prior 12M Minimum", 0, 0),
        ],
    ),
    "Liquidity": (
        "Expanding",
        "Neutral",
        "Contracting",
        1,
        [
            ("M2 Growth 6M Percent", 1, 0.1),
            ("M2 Growth 3M Versus Prior 3M PP", 1, 0.1),
            ("M2 YoY Acceleration 3M PP", 1, 0.05),
        ],
    ),
    "Consumer": (
        "Improving",
        "Neutral",
        "Weakening",
        2,
        [
            ("Consumer_Sentiment Growth 1M Percent", 1, 0.1),
            ("Consumer_Sentiment Growth 6M Percent", 1, 0.1),
            ("Consumer_Sentiment Growth 3M Versus Prior 3M PP", 1, 0.1),
            ("Retail_Sales Growth 3M Percent", 1, 0.1),
        ],
    ),
    "Housing": (
        "Improving",
        "Neutral",
        "Weakening",
        2,
        [
            ("Housing_Starts Growth 1M Percent", 1, 0.1),
            ("Housing_Starts Growth 6M Percent", 1, 0.1),
            ("Housing_Starts Growth 3M Versus Prior 3M PP", 1, 0.1),
            ("Housing_Starts YoY Acceleration 3M PP", 1, 0.05),
        ],
    ),
}


def confirm_regimes(candidates, confirmation_days=5):
    if (
        isinstance(confirmation_days, bool)
        or int(confirmation_days) != confirmation_days
        or confirmation_days < 1
    ):
        raise ValueError("confirmation_days must be a positive integer")
    active, pending, count, values = "Unknown", None, 0, []
    for candidate in candidates:
        if candidate == "Unknown":
            pending, count = None, 0
        elif active == "Unknown":
            active = candidate
        elif candidate == active:
            pending, count = None, 0
        else:
            count = count + 1 if pending == candidate else 1
            pending = candidate
            if count >= confirmation_days:
                active, pending, count = candidate, None, 0
        values.append(active)
    return values


def build_macro_regimes(macro, confirmation_days=5, *, return_daily=False):
    data = macro.copy()
    if "Date" not in data:
        data = data.rename_axis("Date").reset_index()
    data["Date"] = pd.to_datetime(data.Date).dt.tz_localize(None).dt.normalize()
    data = data.sort_values("Date").reset_index(drop=True)
    if data.Date.isna().any() or data.Date.duplicated().any():
        raise ValueError("Macro dates must be unique and nonmissing")
    daily = data[["Date"]].copy()
    regimes = {}
    for family, (positive, neutral, negative, minimum, definitions) in FAMILIES.items():
        votes = pd.DataFrame(index=data.index)
        for name, sign, threshold in definitions:
            x = pd.to_numeric(
                data.get(PREFIX + name, pd.Series(np.nan, index=data.index)), errors="coerce"
            )
            x = x.replace([np.inf, -np.inf], np.nan)
            v = pd.Series(np.nan, index=x.index)
            if sign == 0:
                v.loc[x.notna()] = np.where(
                    x[x.notna()] < 0.2, 1, np.where(x[x.notna()] < 0.5, 0, -1)
                )
            else:
                y = x * sign
                v.loc[x.notna()] = np.where(
                    y[x.notna()] > threshold, 1, np.where(y[x.notna()] < -threshold, -1, 0)
                )
            votes[name] = v
        count, score = votes.count(axis=1), votes.mean(axis=1)
        candidate = pd.Series(
            np.where(score >= 0.34, positive, np.where(score <= -0.34, negative, neutral))
        )
        candidate.loc[count < minimum] = "Unknown"
        daily[family] = confirm_regimes(candidate, confirmation_days)
        daily[family + " Score"] = score
        daily[family + " Available Signals"] = count
        regimes[family] = {name: [] for name in [positive, neutral, negative, "Unknown"]}
        labels = daily[family]
        for _, episode in daily.groupby(labels.ne(labels.shift()).cumsum(), sort=False):
            regimes[family][episode[family].iloc[0]].append(
                (
                    episode.Date.iloc[0].strftime("%Y-%m-%d"),
                    episode.Date.iloc[-1].strftime("%Y-%m-%d"),
                )
            )
    return (regimes, daily) if return_daily else regimes


def macro_database(path=None):
    if path:
        result = Path(path)
    else:
        candidates = [p for p in data_root().glob("*.db") if p.name.lower() == "macro_data.db"]
        if len(candidates) > 1:
            raise ValueError("Multiple macro databases; specify one explicitly")
        result = candidates[0] if candidates else data_root() / "macro_data.db"
    return result


def save_macro_regimes(database=None, table=None, output_dir=None, confirmation_days=5):
    path = macro_database(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        eligible = []
        for name in names:
            escaped = '"' + name.replace('"', '""') + '"'
            columns = [row[1] for row in connection.execute("PRAGMA table_info(" + escaped + ")")]
            if "Date" in columns and any(c.startswith(PREFIX) for c in columns):
                eligible.append(name)
        if table is None:
            if len(eligible) != 1:
                raise ValueError(f"Specify macro table explicitly; PIT tables: {eligible}")
            table = eligible[0]
        if table not in eligible:
            raise ValueError("Selected table must contain Date and Macro PIT features")
        data = pd.read_sql_query('SELECT * FROM "' + table.replace('"', '""') + '"', connection)
    regimes, daily = build_macro_regimes(data, confirmation_days, return_daily=True)
    output = Path(output_dir or path.parent)
    output.mkdir(parents=True, exist_ok=True)
    (output / "Macro_Regimes.txt").write_text(pformat(regimes, sort_dicts=False, width=100) + "\n")
    daily.to_csv(output / "Macro_Regimes_Daily.csv", index=False)
    return regimes


def load_macro_regimes(path):
    value = ast.literal_eval(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected regime dictionary")
    return value


def align_macro_features(stocks, macro):
    """Attach only already-available PIT rows by backward date join.

    Date denotes availability, not economic observation date. Existing selected
    features are retained; this function never overwrites caller columns.
    """
    result = stocks.copy()
    columns = [c for c in macro if c.startswith(PREFIX) and c not in result]
    if not columns:
        return result
    right = macro[["Date", *columns]].copy()
    right["Date"] = pd.to_datetime(right.Date).dt.tz_localize(None).dt.normalize()
    if right.Date.isna().any() or right.Date.duplicated().any():
        raise ValueError("Macro dates must be unique and nonmissing")
    left = pd.DataFrame(
        {
            "Date": pd.to_datetime(result.Date).dt.tz_localize(None).dt.normalize().to_numpy(),
            "_row": np.arange(len(result)),
        }
    )
    if left.Date.isna().any():
        raise ValueError("Stock dates must be nonmissing")
    aligned = pd.merge_asof(
        left.sort_values("Date"), right.sort_values("Date"), on="Date", direction="backward"
    ).sort_values("_row")
    result[columns] = aligned[columns].to_numpy()
    return result


def attach_existing_macro(stocks, table=None):
    """Optional stock-data enrichment from the existing shared macro database."""
    path = macro_database()
    if not path.is_file():
        return stocks
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        candidates = []
        for name in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            columns = [row[1] for row in connection.execute("PRAGMA table_info(" + quoted + ")")]
            if "Date" in columns and any(c.startswith(PREFIX) for c in columns):
                candidates.append(name)
        if table not in candidates:
            if len(candidates) != 1:
                raise ValueError(f"Specify a macro table matching STOCK_TYPE: {candidates}")
            table = candidates[0]
        macro = pd.read_sql_query('SELECT * FROM "' + table.replace('"', '""') + '"', connection)
    return align_macro_features(stocks, macro)

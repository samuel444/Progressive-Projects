"""Read-only, evidence-limited checks of Equity Selector SQLite artifacts."""

import argparse
from collections import Counter
import hashlib
import logging
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .database import quote_identifier
from .parameters import parameters_to_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_databases(paths, *, max_rows=1000000, max_cells=5000000, contract=None):
    """Contract maps database basename -> table -> expected schema/dates/row count.

    No pickle/eval or data writes. Missing evidence produces WARN, never a verified pass.
    Stop writers before auditing: SQL uses a snapshot and hashes detect file changes.
    """
    report = {
        "checks": [],
        "databases": [],
        "limitations": [
            "Stored outputs cannot prove point-in-time input availability or untouched holdouts.",
            "Metrics without underlying predictions/returns cannot be independently recomputed.",
            "Stop database writers for a reproducible audit; WAL files belong to the database.",
        ],
    }
    frames = {}
    contract = contract or {}

    def check(status, code, database, table, message, count=None):
        item = dict(status=status, code=code, database=database, table=table, message=message)
        if count is not None:
            item["count"] = int(count)
        report["checks"].append(item)

    for supplied in paths:
        path = Path(supplied).expanduser().resolve()
        db = str(path)
        logging.getLogger(__name__).info("Checking database %s", path.name)
        if not path.is_file():
            check("FAIL", "missing_database", db, None, "Database does not exist; nothing created.")
            continue
        before = digest(path)
        record = {"path": db, "bytes": path.stat().st_size, "sha256": before, "tables": {}}
        report["databases"].append(record)
        if not path.stat().st_size:
            check("FAIL", "empty_database", db, None, "Zero-byte file contains no results.")
            continue
        try:
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as con:
                con.execute("PRAGMA query_only=ON")
                con.execute("BEGIN")
                integrity = [r[0] for r in con.execute("PRAGMA quick_check")]
                check(
                    "PASS" if integrity == ["ok"] else "FAIL",
                    "sqlite_integrity",
                    db,
                    None,
                    str(integrity),
                )
                fk = list(con.execute("PRAGMA foreign_key_check"))
                check(
                    "FAIL" if fk else "PASS",
                    "foreign_keys",
                    db,
                    None,
                    "SQLite foreign-key violations.",
                    len(fk),
                )
                tables = [
                    r[0]
                    for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                ]
                if not tables:
                    check("FAIL", "no_tables", db, None, "No result tables found.")
                expected = {
                    "Backtest_Database.db": ["Market", "Stocks"],
                    "Portfolio_Simulation_Results.db": [
                        "Stock Simulation Results",
                        "Market Simulation Results",
                    ],
                }.get(path.name, [])
                expected += [
                    name for name in contract.get(path.name, {}) if not name.startswith("_")
                ]
                for table in dict.fromkeys(expected):
                    if table not in tables:
                        check("FAIL", "missing_table", db, table, "Expected table is missing.")
                for table in tables:
                    q = quote_identifier(table)
                    count = con.execute(f"SELECT COUNT(*) FROM {q}").fetchone()[0]
                    columns = [r[1] for r in con.execute(f"PRAGMA table_info({q})")]
                    record["tables"][table] = {"rows": count, "columns": columns}
                    if count > max_rows or count * len(columns) > max_cells:
                        check(
                            "WARN",
                            "table_not_scanned",
                            db,
                            table,
                            f"Table has {count} rows x {len(columns)} columns; exceeds max_rows={max_rows} or max_cells={max_cells}. Increase limits only with sufficient memory.",
                        )
                        continue
                    frame = pd.read_sql_query(f"SELECT * FROM {q}", con)
                    frames[(db, table)] = frame
                    rule = contract.get(path.name, {}).get(table, {})
                    missing = set(rule.get("required_columns", [])) - set(columns)
                    if missing:
                        check("FAIL", "missing_columns", db, table, str(sorted(missing)))
                    if "expected_rows" in rule:
                        check(
                            "PASS" if count == rule["expected_rows"] else "FAIL",
                            "expected_rows",
                            db,
                            table,
                            f"Actual={count}; expected={rule['expected_rows']}",
                        )
                    if not count:
                        check(
                            "WARN",
                            "empty_table",
                            db,
                            table,
                            "Empty may mean disqualification or an incomplete run; inspect stage logs.",
                        )
                        continue
                    known = False
                    if "Date" in frame:
                        known = True
                        dates = pd.to_datetime(frame.Date, errors="coerce", utc=True)
                        invalid = int(dates.isna().sum())
                        check(
                            "FAIL" if invalid else "PASS",
                            "dates_parse",
                            db,
                            table,
                            "Unparseable or missing dates.",
                            invalid,
                        )
                        if dates.notna().any():
                            record["tables"][table]["date_min"] = str(dates.min())
                            record["tables"][table]["date_max"] = str(dates.max())
                        if "date_min" in rule or "date_max" in rule:
                            bad = dates.isna()
                            if "date_min" in rule:
                                bad |= dates < pd.to_datetime(rule["date_min"], utc=True)
                            if "date_max" in rule:
                                bad |= dates > pd.to_datetime(rule["date_max"], utc=True)
                            check(
                                "FAIL" if bad.any() else "PASS",
                                "date_bounds",
                                db,
                                table,
                                "Rows outside declared inclusive timestamp bounds.",
                                bad.sum(),
                            )
                    keys = rule.get("unique_keys")
                    if keys is None and "Date" in frame:
                        keys = [
                            c
                            for c in [
                                "Date",
                                "Simulation ID",
                                "Ticker",
                                "Portfolio Target Type",
                                "Horizon Key",
                            ]
                            if c in frame
                        ]
                    if keys:
                        if set(keys) - set(columns):
                            check("FAIL", "missing_key_columns", db, table, str(keys))
                        else:
                            bad = frame.duplicated(keys, keep=False) | frame[keys].isna().any(
                                axis=1
                            )
                            check(
                                "FAIL" if bad.any() else "PASS",
                                "row_identity",
                                db,
                                table,
                                f"Duplicate/null key rows: {keys}",
                                bad.sum(),
                            )
                    if {"Date", "Ticker", "Return"} <= set(frame):
                        conflicts = (
                            frame.groupby(["Date", "Ticker"], dropna=False)
                            .Return.nunique(dropna=False)
                            .gt(1)
                            .sum()
                        )
                        check(
                            "FAIL" if conflicts else "PASS",
                            "return_alignment",
                            db,
                            table,
                            "Same date/ticker must have one return across target/horizon rows.",
                            conflicts,
                        )
                    if {"Model", "Parameters"} <= set(frame):
                        known = True
                        canonical = []
                        errors = 0
                        for value in frame.Parameters:
                            try:
                                canonical.append(parameters_to_json(value))
                            except (ValueError, TypeError, SyntaxError):
                                canonical.append(None)
                                errors += 1
                        frame["_parameters_key"] = canonical
                        check(
                            "FAIL" if errors else "PASS",
                            "parameters_parse",
                            db,
                            table,
                            "Unsafe/malformed parameter dictionaries.",
                            errors,
                        )
                        identity = [
                            c
                            for c in ["Target", "Model", "_parameters_key", "Fold"]
                            if c in frame and (c != "Fold" or table.endswith("__folds"))
                        ]
                        duplicates = (
                            frame.loc[frame._parameters_key.notna()]
                            .duplicated(identity, keep=False)
                            .sum()
                        )
                        check(
                            "FAIL" if duplicates else "PASS",
                            "model_identity",
                            db,
                            table,
                            "Duplicate canonical model configurations (within fold for raw folds).",
                            duplicates,
                        )
                    if "Simulation ID" in frame and "Date" not in frame:
                        known = True
                        bad = frame["Simulation ID"].isna() | frame["Simulation ID"].duplicated(
                            keep=False
                        )
                        check(
                            "FAIL" if bad.any() else "PASS",
                            "simulation_identity",
                            db,
                            table,
                            "Simulation IDs must be unique and non-null.",
                            bad.sum(),
                        )
                    ranges = {
                        "Return": (-1, None),
                        "Strategy Return": (-1, None),
                        "Max Drawdown": (-1, 0),
                        "Average Drawdown": (-1, 0),
                        "Rank IC": (-1, 1),
                        "ROC AUC": (0, 1),
                        "PR AUC": (0, 1),
                        "F1": (0, 1),
                        "Macro F1": (0, 1),
                        "Balanced Accuracy": (0, 1),
                        "RMSE": (0, None),
                        "NRMSE": (0, None),
                        "MAE": (0, None),
                        "Log Loss": (0, None),
                        "Horizon": (1, None),
                        "Max Weight": (0, 1),
                    }
                    for col in frame:
                        base = col.removesuffix(" Mean")
                        bounds = (0, None) if col.endswith(" Std") else ranges.get(base)
                        if bounds is None and col != "Signal":
                            continue
                        numeric = pd.to_numeric(frame[col], errors="coerce")
                        bad = (frame[col].notna() & numeric.isna()) | np.isinf(numeric)
                        if bounds:
                            lo, hi = bounds
                            if lo is not None:
                                bad |= numeric < lo - 1e-10
                            if hi is not None:
                                bad |= numeric > hi + 1e-10
                        if bad.any():
                            check(
                                "FAIL",
                                "metric_domain",
                                db,
                                table,
                                f"{col}: nonnumeric, infinite or out-of-domain values.",
                                bad.sum(),
                            )
                    if "Signal" in frame and frame.Signal.isna().any():
                        check(
                            "WARN",
                            "missing_signal",
                            db,
                            table,
                            "Missing signals require disqualification/coverage interpretation.",
                            frame.Signal.isna().sum(),
                        )
                    if table.endswith("__folds"):
                        if "Fold" in frame:
                            fold = pd.to_numeric(frame.Fold, errors="coerce")
                            bad = fold.isna() | (fold < 1) | (fold % 1 != 0)
                            check(
                                "FAIL" if bad.any() else "PASS",
                                "fold_numbers",
                                db,
                                table,
                                "Fold numbers must be positive integers.",
                                bad.sum(),
                            )
                        if {"Train End", "Validation Start", "Validation End"} <= set(frame):
                            te = pd.to_datetime(frame["Train End"], errors="coerce", utc=True)
                            vs = pd.to_datetime(
                                frame["Validation Start"], errors="coerce", utc=True
                            )
                            ve = pd.to_datetime(frame["Validation End"], errors="coerce", utc=True)
                            bad = te.isna() | vs.isna() | ve.isna() | (te >= vs) | (vs > ve)
                            check(
                                "FAIL" if bad.any() else "PASS",
                                "fold_chronology",
                                db,
                                table,
                                "Training must precede validation.",
                                bad.sum(),
                            )
                        else:
                            check(
                                "WARN",
                                "fold_dates_unverifiable",
                                db,
                                table,
                                "Fold timestamps/purge evidence are absent; boundaries cannot be verified.",
                            )
                    if not known:
                        check(
                            "WARN",
                            "unrecognized_schema",
                            db,
                            table,
                            "Only structural/declared-contract checks apply; no automatic semantic validator.",
                        )
                con.rollback()
        except (sqlite3.Error, pd.errors.DatabaseError) as error:
            check("FAIL", "sqlite_read", db, None, str(error))
        if digest(path) != before:
            check(
                "FAIL",
                "file_changed",
                db,
                None,
                "Database changed during audit; stop writers and retry.",
            )
        else:
            check("PASS", "source_unchanged", db, None, "SHA-256 unchanged after read-only scan.")
        if Path(db + "-wal").exists():
            check(
                "WARN",
                "wal_present",
                db,
                None,
                "WAL present: main-file hash alone does not identify all database content. Transfer a SQLite backup, not just the main file.",
            )

    # Independent aggregation checks: saved search summaries must agree with raw folds.
    for (db, table), summary in list(frames.items()):
        if not table.endswith("__search") or summary.empty or "_parameters_key" not in summary:
            continue
        folds = frames.get((db, table.removesuffix("__search") + "__folds"))
        if folds is None or folds.empty or "_parameters_key" not in folds:
            check(
                "WARN", "summary_unverifiable", db, table, "Raw fold results missing/not scanned."
            )
            continue
        group = folds.groupby(["Model", "_parameters_key"], dropna=False)
        mismatches = missing = 0
        for _, row in summary.iterrows():
            key = (row.Model, row._parameters_key)
            if key not in group.groups:
                missing += 1
                continue
            raw = group.get_group(key)
            for column in summary:
                suffix = (
                    " Mean"
                    if column.endswith(" Mean")
                    else " Std"
                    if column.endswith(" Std")
                    else None
                )
                if suffix and column.removesuffix(suffix) in raw:
                    values = pd.to_numeric(raw[column.removesuffix(suffix)], errors="coerce")
                    expected = values.mean() if suffix == " Mean" else values.std(ddof=1)
                    actual = pd.to_numeric(row[column], errors="coerce")
                    if not np.isclose(actual, expected, equal_nan=True, rtol=1e-6, atol=1e-9):
                        mismatches += 1
        check(
            "FAIL" if missing or mismatches else "PASS",
            "fold_summary_replay",
            db,
            table,
            f"{missing} missing model groups; {mismatches} mean/std cells disagree.",
            missing + mismatches,
        )

    for (db, table), stock in frames.items():
        if table == "Stock Simulation Results":
            market = frames.get((db, "Market Simulation Results"))
            if market is not None and "Simulation ID" in stock and "Simulation ID" in market:
                same = set(stock["Simulation ID"]) == set(market["Simulation ID"])
                check(
                    "PASS" if same else "FAIL",
                    "simulation_id_alignment",
                    db,
                    table,
                    "Strategy and market must cover the same IDs; row order is irrelevant.",
                )
        if table.endswith(" Passed Test Results") and {"Target", "Model", "_parameters_key"} <= set(
            stock
        ):
            original = frames.get(
                (db, "Final Test Results " + table.removesuffix(" Passed Test Results"))
            )
            if original is not None and "_parameters_key" in original:
                keys = ["Target", "Model", "_parameters_key"]
                candidates = set(map(tuple, original[keys].to_numpy()))
                absent = sum(tuple(row) not in candidates for row in stock[keys].to_numpy())
                check(
                    "FAIL" if absent else "PASS",
                    "passed_model_membership",
                    db,
                    table,
                    "Passed models must occur in final-test results.",
                    absent,
                )
    # Optional return-to-summary replay uses independent arithmetic, not the production metric helper.
    for db in {key[0] for key in frames}:
        for replay in contract.get(Path(db).name, {}).get("_metric_replays", []):
            daily_name, summary_name = replay["daily_table"], replay["summary_table"]
            daily, summary = frames.get((db, daily_name)), frames.get((db, summary_name))
            if daily is None or summary is None:
                check(
                    "WARN",
                    "portfolio_replay_unavailable",
                    db,
                    summary_name,
                    "Daily/summary table not available or not scanned.",
                )
                continue
            group_col = replay.get("group_column")
            annualisation = replay.get("annualisation", 252)
            if not np.isfinite(annualisation) or annualisation <= 0:
                raise ValueError("Replay annualisation must be positive and finite")
            groups = daily.groupby(group_col) if group_col else [(None, daily)]
            mismatches = 0
            for identity, observations in groups:
                target = summary.loc[summary[group_col].eq(identity)] if group_col else summary
                if len(target) != 1:
                    mismatches += 1
                    continue
                observations = observations.copy()
                observations["Date"] = pd.to_datetime(observations.Date, errors="coerce", utc=True)
                if observations.Date.isna().any() or observations.Date.duplicated().any():
                    mismatches += 1
                    continue
                r = pd.to_numeric(
                    observations.sort_values("Date")[replay.get("return_column", "Return")],
                    errors="coerce",
                ).to_numpy()
                if len(r) < 2 or not np.isfinite(r).all() or (r < -1).any():
                    mismatches += 1
                    continue
                nav = np.cumprod(1 + r)
                peaks = np.maximum.accumulate(np.r_[1.0, nav])[1:]
                drawdowns = nav / peaks - 1
                std = np.std(r, ddof=1)
                expected = {
                    "Return": nav[-1] - 1,
                    "Volatility": std * np.sqrt(annualisation),
                    "Sharpe Ratio": np.mean(r) / std * np.sqrt(annualisation)
                    if std > 0
                    else np.nan,
                    "Max Drawdown": drawdowns.min(),
                    "Average Drawdown": drawdowns.mean(),
                }
                mapping = replay.get("metrics", {name: name for name in expected})
                for metric, column in mapping.items():
                    if metric not in expected or column not in target:
                        mismatches += 1
                        continue
                    actual = pd.to_numeric(target[column], errors="coerce").iloc[0]
                    if not np.isclose(
                        actual, expected[metric], rtol=1e-6, atol=1e-9, equal_nan=True
                    ):
                        mismatches += 1
            check(
                "FAIL" if mismatches else "PASS",
                "portfolio_metric_replay",
                db,
                summary_name,
                "Independent compounded return, sample volatility, Sharpe and initial-NAV drawdown replay.",
                mismatches,
            )
    counts = Counter(item["status"] for item in report["checks"])
    report["counts"] = dict(counts)
    report["status"] = "FAIL" if counts["FAIL"] else "INCOMPLETE" if counts["WARN"] else "PASS"
    report["meaning"] = (
        "PASS means only the listed checks passed; it is not a guarantee of correct research or profitability."
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path, help="SQLite files to inspect (or use --data-dir)."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--report", type=Path, default=Path("database-audit.json"))
    parser.add_argument(
        "--contract",
        type=Path,
        help="Optional JSON with expected tables, columns, keys, bounds and row counts.",
    )
    parser.add_argument("--max-rows", type=int, default=1000000)
    parser.add_argument("--max-cells", type=int, default=5000000)
    args = parser.parse_args(argv)
    paths = args.paths + (sorted(args.data_dir.rglob("*.db")) if args.data_dir else [])
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if not paths:
        parser.error("Supply database paths or a data directory containing .db files.")
    if args.max_rows <= 0 or args.max_cells <= 0:
        parser.error("--max-rows and --max-cells must be positive")
    if args.report.resolve() in paths:
        parser.error("Report must not overwrite an input database.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = audit_databases(
        paths,
        max_rows=args.max_rows,
        max_cells=args.max_cells,
        contract=json.loads(args.contract.read_text()) if args.contract else None,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {report['counts']}; report: {args.report.resolve()}")
    return 1 if report["status"] == "FAIL" else 2 if report["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())

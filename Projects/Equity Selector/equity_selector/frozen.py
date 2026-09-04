"""Evaluate every preselected portfolio on a separate period without further selection."""

import ast
import logging
from time import perf_counter
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .account import gbp_account_returns
from .database import read_table, write_frame
from .database_audit import digest
from .metrics import performance_metrics
from .portfolio import portfolio_returns_from_scores
from .scoring import (
    add_horizon_scores,
    add_type_scores,
    apply_horizon_signal_refresh,
    build_score_stocks_with_direction,
)


def evaluate_frozen(
    *,
    selection_database,
    cache_database,
    horizon_file,
    type_configurations,
    start,
    end,
    trading_fee=0.001,
    annualisation=252,
    output_dir,
    account=None,
    evaluation_kind="final",
):
    """Requires already-frozen model/horizon/portfolio choices and separate cache.

    Final mode never selects again; selection mode applies the GBP drawdown gate.
    No model fitting or parameter search. The caller must establish the
    cache's model-training provenance: these result tables alone cannot prove it.
    """
    started = perf_counter()
    logger = logging.getLogger(__name__)
    if evaluation_kind not in {"selection", "final"}:
        raise ValueError("evaluation_kind must be selection or final")
    account = dict(account) if account is not None else None
    if account is not None:
        limit = account.get("max_drawdown", 0.2)
        if not np.isfinite(limit) or not 0 < limit < 1:
            raise ValueError("max_drawdown must lie in (0, 1)")
        execution_cost = account.get("execution_cost_fraction", 0.0)
        if not np.isfinite(execution_cost) or execution_cost < 0:
            raise ValueError("execution_cost_fraction must be nonnegative")
    elif evaluation_kind == "selection":
        raise ValueError("Selection account check requires account settings")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Final start/end must be valid and strictly increasing")
    paths = [
        Path(selection_database).resolve(),
        Path(cache_database).resolve(),
        Path(horizon_file).resolve(),
    ]
    if evaluation_kind == "final" and paths[0] == paths[1]:
        raise ValueError("Final cache must be separate from the selection database")
    out = Path(output_dir).resolve()
    output_db = out / (
        "GBP_Selection.db" if evaluation_kind == "selection" else "Frozen_Final_Evaluation.db"
    )
    if output_db.exists():
        raise FileExistsError("Use a fresh output directory; preserve previous final evaluations")
    if output_db in paths:
        raise ValueError("Output must not overwrite inputs")
    if account is not None:
        paths.append(Path(account["fx_file"]).resolve())
    if output_db in paths:
        raise ValueError("Output must not overwrite inputs")
    hashes = {str(p): digest(p) for p in paths}
    selected = read_table(paths[0], "Passed Strategies")
    source_stocks = read_table(paths[0], "Stocks")
    selection_end = pd.to_datetime(source_stocks.Date, errors="raise").max()
    if evaluation_kind == "final" and (pd.isna(selection_end) or selection_end >= start):
        raise ValueError("Final period must start after all dates in the selection cache")
    required = {
        "Simulation ID",
        "Horizon Score Index",
        "Type Configuration",
        "Rebalance Multiplier",
        "Max Weight",
        "Concentration Penalty",
    }
    if (
        selected.empty
        or not required <= set(selected)
        or selected["Simulation ID"].isna().any()
        or selected["Simulation ID"].duplicated().any()
    ):
        raise ValueError("A nonempty frozen selection with unique IDs and all settings is required")
    configurations = ast.literal_eval(paths[2].read_text())
    if not isinstance(configurations, list) or not all(isinstance(c, dict) for c in configurations):
        raise ValueError("Horizon file must contain a list of dictionaries")
    stocks = read_table(paths[1], "Stocks")
    stocks["Date"] = pd.to_datetime(stocks.Date, errors="raise")
    # Do not silently conceal accidental overlap/extra periods in a purported final cache.
    if stocks.empty or stocks.Date.isna().any() or not stocks.Date.between(start, end).all():
        raise ValueError("Final cache dates must lie entirely within the declared final period")
    summaries = []
    daily = []
    logger.info("Account evaluation | phase=%s candidates=%d", evaluation_kind, len(selected))
    for number, (_, row) in enumerate(selected.iterrows(), 1):
        logger.info("Evaluating candidate %d/%d", number, len(selected))
        index = float(row["Horizon Score Index"])
        if not np.isfinite(index) or not index.is_integer() or not 0 <= index < len(configurations):
            raise ValueError("Invalid frozen Horizon Score Index")
        groups = [c for c in type_configurations if c["Name"] == row["Type Configuration"]]
        if len(groups) != 1:
            raise ValueError("Type configuration must resolve uniquely")
        predictions = add_horizon_scores(stocks.copy(), configurations[int(index)])
        predictions = add_type_scores(predictions, groups[0].copy())
        predictions["Contribution"] = (
            predictions["Horizon Score"] * predictions["Signal"] * predictions["Type Score"]
        )
        predictions = apply_horizon_signal_refresh(predictions, float(row["Rebalance Multiplier"]))
        scores = build_score_stocks_with_direction(predictions)
        frame = portfolio_returns_from_scores(
            scores,
            max_weight=float(row["Max Weight"]),
            concentration_penalty=float(row["Concentration Penalty"]),
            trading_fee=trading_fee
            + (account.get("execution_cost_fraction", 0.0) if account else 0.0),
        )
        if len(frame) < 2:
            raise ValueError("At least two realized final returns are required")
        if account is not None:
            frame = gbp_account_returns(
                frame,
                initial_date=stocks.Date.min(),
                **{
                    k: v
                    for k, v in account.items()
                    if k not in {"max_drawdown", "execution_cost_fraction"}
                },
            )
        frame["Simulation ID"] = row["Simulation ID"]
        daily.append(frame)
        summaries.append(
            {
                "Simulation ID": row["Simulation ID"],
                **performance_metrics(frame.Return, annualisation),
            }
        )
    summary = pd.DataFrame(summaries)
    if account is not None:
        summary["GBP Drawdown Limit Passed"] = summary["Max Drawdown"] >= -account.get(
            "max_drawdown", 0.2
        )
    returns = pd.concat(daily, ignore_index=True)
    for path, before in hashes.items():
        if digest(path) != before:
            raise ValueError("Input changed during evaluation; stop writers and retry")
    out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output_db) as connection:
        write_frame(returns, "Daily Returns", connection, if_exists="replace")
        write_frame(summary, "Summary", connection, if_exists="replace")
        write_frame(selected, "Frozen Settings", connection, if_exists="replace")
        if evaluation_kind == "selection":
            ids = summary.loc[summary["GBP Drawdown Limit Passed"], "Simulation ID"]
            write_frame(
                selected.loc[selected["Simulation ID"].isin(ids)],
                "Passed Strategies",
                connection,
                if_exists="replace",
            )
            write_frame(
                stocks[["Date"]].drop_duplicates(), "Stocks", connection, if_exists="replace"
            )
    summary.to_csv(out / "Frozen_Final_Summary.csv", index=False)
    (out / "Frozen_Final_Manifest.json").write_text(
        json.dumps(
            {
                "input_sha256": hashes,
                "selection_cache_end": str(selection_end),
                "start": str(start),
                "end": str(end),
                "actual_first_return": str(returns.Date.min()),
                "actual_last_return": str(returns.Date.max()),
                "trading_fee_fraction": trading_fee,
                "annualisation": annualisation,
                "type_configurations": type_configurations,
                "selection_applied": evaluation_kind == "selection",
                "evaluation_kind": evaluation_kind,
                "account": account,
                "return_currency": "GBP" if account else "USD",
                "limitations": [
                    "Cached model training provenance must be independently established.",
                    "No claim that previously inspected dates are truly unseen.",
                    "First cache date supplies the signal; its return is not counted.",
                ],
            },
            indent=2,
        )
    )
    contract = {
        output_db.name: {
            "_metric_replays": [
                {
                    "daily_table": "Daily Returns",
                    "summary_table": "Summary",
                    "group_column": "Simulation ID",
                    "annualisation": annualisation,
                }
            ]
        }
    }
    (out / "audit-contract.json").write_text(json.dumps(contract, indent=2))
    logger.info(
        "Evaluation complete | candidates=%d elapsed=%.1fs", len(summary), perf_counter() - started
    )
    return summary

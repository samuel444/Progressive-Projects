from equity_selector.settings import setting as get_setting
from equity_selector.config import data_root

"Standalone Equity Selector final evaluation.\n\nRun this file directly. It reads Passed Strategies and source data from SQLite.\nThe simulation script and main_package are never imported or executed.\nCopied helper definitions preserve the supplied backtest conventions.\nDependencies: numpy, pandas. See --help for paths and evaluation assumptions.\n"
from equity_selector.database import write_frame
import argparse
import ast
import json
import copy
import logging
import sqlite3
from statistics import NormalDist
from pathlib import Path
import numpy as np
import pandas as pd

PORTFOLIO_RANKING_TYPES = {"ALPHA", "RELATIVE_ALPHA", "RISK_ADJUSTED_ALPHA", "CROSS_SECTION_ALPHA"}
PORTFOLIO_DIRECTION_TYPES = {"DIRECTION", "DIRECTION_MULTICLASS", "ALPHA_BINARY", "BARRIER_ALPHA"}
PORTFOLIO_RISK_TYPES = {
    "VOLATILITY",
    "DOWNSIDE_VOLATILITY",
    "VOLATILITY_ASYMMETRY",
    "DOWNSIDE",
    "TAIL_RISK",
    "TAIL_EVENT",
    "DOWNSIDE_EXCURSION",
    "VOLATILITY_EVENT",
    "CROSS_SECTION_DOWNSIDE",
}
PORTFOLIO_OPPORTUNITY_TYPES = {
    "ABSOLUTE_MOVE",
    "UPSIDE_VOLATILITY",
    "UPSIDE_EVENT",
    "UPSIDE_EXCURSION",
    "RECOVERY",
    "REVERSAL",
}
PORTFOLIO_SPECIAL_TYPES = {
    "TIME_TO_DOWNSIDE_EXCURSION",
    "TIME_TO_UPSIDE_EXCURSION",
    "EXECUTION",
    "LIQUIDITY",
    "MARKET_IMPACT",
    "CORRELATION",
    "COVARIANCE",
    "REGIME",
}
PORTFOLIO_GROUP_CONFIGURATIONS = [
    {
        "Name": "Balanced",
        "Ranking": 0.3,
        "Direction": 0.25,
        "Risk": 0.25,
        "Opportunity": 0.15,
        "Special": 0.05,
    },
    {
        "Name": "Equal Weight",
        "Ranking": 0.2,
        "Direction": 0.2,
        "Risk": 0.2,
        "Opportunity": 0.2,
        "Special": 0.2,
    },
    {
        "Name": "Core Balanced",
        "Ranking": 0.35,
        "Direction": 0.3,
        "Risk": 0.25,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Ranking Heavy",
        "Ranking": 0.5,
        "Direction": 0.2,
        "Risk": 0.2,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Ranking And Risk",
        "Ranking": 0.45,
        "Direction": 0.15,
        "Risk": 0.3,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Ranking And Direction",
        "Ranking": 0.45,
        "Direction": 0.3,
        "Risk": 0.15,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Direction Heavy",
        "Ranking": 0.25,
        "Direction": 0.45,
        "Risk": 0.2,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Direction And Risk",
        "Ranking": 0.25,
        "Direction": 0.4,
        "Risk": 0.3,
        "Opportunity": 0.05,
        "Special": 0.0,
    },
    {
        "Name": "Risk Heavy",
        "Ranking": 0.25,
        "Direction": 0.2,
        "Risk": 0.45,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Conservative",
        "Ranking": 0.3,
        "Direction": 0.15,
        "Risk": 0.45,
        "Opportunity": 0.05,
        "Special": 0.05,
    },
    {
        "Name": "Opportunity Heavy",
        "Ranking": 0.25,
        "Direction": 0.2,
        "Risk": 0.2,
        "Opportunity": 0.35,
        "Special": 0.0,
    },
    {
        "Name": "Ranking And Special",
        "Ranking": 0.4,
        "Direction": 0.15,
        "Risk": 0.2,
        "Opportunity": 0.1,
        "Special": 0.15,
    },
]
from equity_selector.scoring import build_score_stocks_with_direction
from equity_selector.scoring import apply_horizon_signal_refresh
from equity_selector.scoring import add_horizon_scores
from equity_selector.scoring import add_type_scores


def result_metrics(results_dataframe):
    from equity_selector.metrics import relative_metrics

    return relative_metrics(
        results_dataframe,
        {
            "Return": market_return,
            "Sharpe Ratio": market_sharpe,
            "Max Drawdown": market_max_drawdown,
            "Average Drawdown": market_average_drawdown,
        },
    )


from equity_selector.portfolio import portfolio_returns_from_scores

FE_TRADING_FEE = 0.001
FE_RF_ANNUAL = 0.0
FE_DSR_SAMPLE_SIZE = 100
FE_DSR_TRIALS = None
FE_SEED = 20260904
FE_DAYS = 252
FE_NEIGHBOURHOOD_SD = np.nan
FE_UNSEEN_GATE = 1.5
FE_EXISTING_COLUMNS = {
    "Best Day Removed Quality": "Best Day Removed Quality",
    "Best Week Removed Quality": "Best Week Removed Quality",
    "Best Month Removed Quality": "Best Month Removed Quality",
    "Best Year Removed Quality": "Best Year Removed Quality",
    "Mean Stock Removal Quality": "Mean Stock Removal Quality",
    "Worst Stock Removal Quality": "Worst Stock Removal Quality",
    "Worst Removed Ticker": "Worst Removed Ticker",
    "Neighbourhood Score": "Neighbourhood Score",
    "Neighbourhood Pass Rate": "Neighbourhood Pass Rate",
    "Unseen Stock Score": "Unseen Stock Score",
    "Unseen Backtest Quality": "Unseen Backtest Quality",
    "Unseen Gate Passed": "Unseen Gate Passed",
    "Portfolio Target Score": "Portfolio Target Score",
    "Market Target Score": "Market Target Score",
    "Relative Target Score": "Relative Target Score",
}
FE_METRICS = [
    "Strategy Return",
    "Sharpe Ratio",
    "Average Drawdown",
    "Max Drawdown",
    "Relative Return",
    "Relative Sharpe Ratio",
    "Relative Max Drawdown",
    "Relative Average Drawdown",
    "Backtest Quality",
]
FE_SETTINGS = [
    "Horizon Score Index",
    "Type Configuration",
    "Rebalance Multiplier",
    "Max Weight",
    "Concentration Penalty",
]


def fe_series(frame):
    s = frame.set_index("Date")["Return"].copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().astype(float)
    if s.index.has_duplicates or not np.isfinite(s).all() or (s < -1).any():
        raise ValueError("Invalid, duplicated or missing daily returns.")
    if len(s) < 3:
        raise ValueError("At least three daily observations required.")
    return s


def fe_sharpe(r, annual=True):
    x = np.asarray(r, dtype=float) - ((1 + FE_RF_ANNUAL) ** (1 / FE_DAYS) - 1)
    sd = x.std(ddof=1)
    return x.mean() / sd * (np.sqrt(FE_DAYS) if annual else 1) if sd > 0 else np.nan


def fe_psr(r, benchmark=0.0):
    x = np.asarray(r, dtype=float) - ((1 + FE_RF_ANNUAL) ** (1 / FE_DAYS) - 1)
    n = len(x)
    sd = x.std(ddof=1)
    if n < 30 or sd <= 0 or (not np.isfinite(benchmark)):
        return np.nan
    sr = x.mean() / sd
    centered = x - x.mean()
    m2 = np.mean(centered**2)
    skew = np.mean(centered**3) / m2**1.5
    kurtosis = np.mean(centered**4) / m2**2
    variance = 1 - skew * sr + (kurtosis - 1) * sr**2 / 4
    if variance <= 0:
        return np.nan
    return NormalDist().cdf((sr - benchmark) * np.sqrt(n - 1) / np.sqrt(variance))


def fe_quality(r):
    return dict(
        zip(FE_METRICS, result_metrics(pd.DataFrame({"Date": r.index, "Return": r.to_numpy()})))
    )


def fe_scores(row, universe):
    configs = [c for c in PORTFOLIO_GROUP_CONFIGURATIONS if c["Name"] == row["Type Configuration"]]
    if len(configs) != 1:
        raise ValueError("Type Configuration must identify exactly one configuration.")
    h = float(row["Horizon Score Index"])
    if not h.is_integer() or not 0 <= h < len(horizon_score_configurations):
        raise ValueError("Invalid Horizon Score Index.")
    d = add_horizon_scores(universe.copy(), horizon_score_configurations[int(h)])
    d = add_type_scores(d, copy.deepcopy(configs[0]))
    d["Contribution"] = d["Horizon Score"] * d["Signal"] * d["Type Score"]
    d = apply_horizon_signal_refresh(d, float(row["Rebalance Multiplier"]))
    return build_score_stocks_with_direction(d)


def fe_run_scores(row, scores):
    frame = portfolio_returns_from_scores(
        scores.copy(),
        max_weight=float(row["Max Weight"]),
        concentration_penalty=float(row["Concentration Penalty"]),
        trading_fee=0.0,
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    fe_series(frame)
    return frame


def fe_best_removed(r, window):
    if len(r) <= window:
        return np.nan
    rolling = (1 + r).rolling(window).apply(np.prod, raw=True)
    end = int(np.nanargmax(rolling.to_numpy()))
    changed = r.copy()
    changed.iloc[end - window + 1 : end + 1] = 0.0
    return fe_quality(changed)["Backtest Quality"]


def validated_fee(fee):
    if not np.isfinite(fee) or fee < 0:
        raise ValueError("Trading fee must be finite and nonnegative")
    return float(fee)


def fe_costs(frame):
    w = frame.drop(columns=["Date", "Return"]).astype(float)
    if w.empty or not np.isfinite(w.to_numpy()).all():
        raise ValueError("Daily ticker weights missing or invalid.")
    delta = w.diff()
    delta.iloc[0] = w.iloc[0]
    traded = delta.abs().sum(axis=1).to_numpy()
    r = frame["Return"].to_numpy(dtype=float)
    c = validated_fee(FE_TRADING_FEE)
    if np.any(1 + r - 2 * c * traded <= 0):
        raise ValueError("Cost stress causes insolvency; inspect the daily results.")
    if np.prod(1 + r) <= 1:
        breakeven = 0.0
    elif not np.any(traded > 0):
        breakeven = np.inf
    else:
        low = 0.0
        high = np.min((1 + r[traded > 0]) / traded[traded > 0])
        for _ in range(80):
            mid = (low + high) / 2
            if np.log1p(r - mid * traded).sum() > 0:
                low = mid
            else:
                high = mid
        breakeven = (low + high) / 2 * 10000
    return (
        {
            "Annual Turnover": 0.5 * traded.mean() * FE_DAYS,
            "Sharpe at Realistic Costs": fe_sharpe(r - c * traded),
            "Sharpe at 2x Costs": fe_sharpe(r - 2 * c * traded),
            "Break-Even Cost (bps)": breakeven,
            "Break-Even Cost (%)": breakeven / 100,
        },
        traded,
    )


def fe_es(r):
    x = np.sort(np.asarray(r, dtype=float))
    mass = 0.05 * len(x)
    whole = int(np.floor(mass))
    return (x[:whole].sum() + (mass - whole) * x[whole]) / mass


def build_final_strategy_evaluation(finalists=None):
    log = logging.getLogger("final_evaluation")
    with sqlite3.connect(
        Path(SIMULATION_RESULTS_DATABASE).resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        grid = pd.read_sql_query('SELECT * FROM "Stock Simulation Results"', connection)
        market_grid = pd.read_sql_query('SELECT * FROM "Market Simulation Results"', connection)
        if finalists is None:
            with sqlite3.connect(
                Path(BACKTEST_DATABASE).resolve().as_uri() + "?mode=ro", uri=True
            ) as passed_connection:
                finalists = pd.read_sql_query(
                    'SELECT * FROM "Passed Strategies"', passed_connection
                )
    selected = finalists.copy(deep=True)
    if not 1 <= len(selected) <= 10:
        raise ValueError("Pass the already-selected 1–10 strategies; no automatic truncation.")
    for df in (selected, grid, market_grid):
        if df["Simulation ID"].duplicated().any():
            raise ValueError("Simulation ID must be unique.")
    if not set(FE_SETTINGS).issubset(selected.columns):
        raise ValueError("Selected rows are missing reconstruction settings.")
    if not set(selected["Simulation ID"]).issubset(set(grid["Simulation ID"])):
        raise ValueError("Finalists do not belong to the saved simulation grid.")
    if (
        not np.isfinite(FE_TRADING_FEE)
        or FE_TRADING_FEE < 0
        or FE_RF_ANNUAL <= -1
        or FE_DSR_SAMPLE_SIZE < 2
    ):
        raise ValueError("Invalid final evaluation assumptions.")
    unseen = None
    if unseen is None:
        with sqlite3.connect(
            Path(BACKTEST_DATABASE).resolve().as_uri() + "?mode=ro", uri=True
        ) as connection:
            unseen = pd.read_sql_query('SELECT * FROM "Unseen"', connection)
    unseen = unseen.copy()
    unseen["Date"] = pd.to_datetime(unseen["Date"])
    if unseen.empty or set(unseen["Ticker"]) & set(stocks["Ticker"]):
        raise ValueError("Unseen universe must be nonempty and disjoint from Stocks.")
    neighbour_sd = FE_NEIGHBOURHOOD_SD
    unseen_sd = market_grid["Backtest Quality"].std()
    unseen_gate = FE_UNSEEN_GATE
    benchmark_rows = market_grid.set_index("Simulation ID")
    raw_grid = grid.set_index("Simulation ID")
    details, rows, streams = ({}, [], {})
    for _, row in selected.iterrows():
        sid = row["Simulation ID"]
        log.info("Final evaluation: simulation %s", sid)
        saved = raw_grid.loc[sid]
        neighbour_sd = float(row.get("Neighbourhood SD", FE_NEIGHBOURHOOD_SD))
        unseen_sd = float(row.get("Unseen Quality SD", market_grid["Backtest Quality"].std()))
        unseen_gate = float(row.get("Unseen Threshold", FE_UNSEEN_GATE))
        for key in FE_SETTINGS:
            equal = (
                str(row[key]) == str(saved[key])
                if key == "Type Configuration"
                else np.isclose(float(row[key]), float(saved[key]))
            )
            if not equal:
                raise ValueError(f"{sid}: settings differ from saved grid: {key}")
        score = fe_scores(row, stocks)
        frame = fe_run_scores(row, score)
        r = fe_series(frame)
        metrics = fe_quality(r)
        mismatches = [
            k
            for k in FE_METRICS
            if k in row
            and (not np.isclose(float(row[k]), metrics[k], equal_nan=True, rtol=1e-07, atol=1e-10))
        ]
        if mismatches:
            raise ValueError(
                f"{sid}: rerun differs from saved metrics: {mismatches}. Check data, helpers and benchmark globals before continuing."
            )
        streams[sid] = r
        out = {"Simulation ID": sid, **metrics}
        for key in ["Neighbourhood Score", "Unseen Stock Score"]:
            out[key] = row.get(key, np.nan)
        for label, window in [("Day", 1), ("Week", 5), ("Month", 21), ("Year", 252)]:
            out[f"Best {label} Removed Quality"] = fe_best_removed(r, window)
        periods = []
        ends = sorted(
            set(range(FE_DAYS, len(r) + 1, 21)) | ({len(r)} if len(r) >= FE_DAYS else set())
        )
        for end in ends:
            part = r.iloc[end - FE_DAYS : end]
            periods.append(
                {
                    "Start": part.index[0],
                    "End": part.index[-1],
                    "Backtest Quality": fe_quality(part)["Backtest Quality"],
                }
            )
        out["Worst 252d Quality (fixed benchmark)"] = (
            min((p["Backtest Quality"] for p in periods)) if periods else np.nan
        )
        removals = []
        for ticker in sorted(score["Ticker"].unique()):
            reduced = score.loc[score["Ticker"].ne(ticker)]
            rr = fe_series(fe_run_scores(row, reduced))
            if not rr.index.equals(r.index):
                raise ValueError(f"{sid}: removing {ticker} changes the date coverage.")
            removals.append(
                {"Ticker": ticker, "Backtest Quality": fe_quality(rr)["Backtest Quality"]}
            )
        removal_df = pd.DataFrame(removals)
        valid = removal_df["Backtest Quality"].notna()
        out["Mean Stock Removal Quality"] = (
            removal_df["Backtest Quality"].mean() if valid.all() else np.nan
        )
        worst = removal_df.loc[removal_df["Backtest Quality"].idxmin()] if valid.any() else None
        out["Worst Stock Removal Quality"] = (
            worst["Backtest Quality"] if worst is not None else np.nan
        )
        out["Worst Removed Ticker"] = worst["Ticker"] if worst is not None else None
        rng = np.random.default_rng(np.random.SeedSequence([FE_SEED, int(sid)]))
        neighbours = []
        for iteration in range(30):
            p = row.copy()
            for key in ["Rebalance Multiplier", "Concentration Penalty", "Max Weight"]:
                p[key] = rng.uniform(
                    max(0.0, float(row[key]) - 0.05), min(1.0, float(row[key]) + 0.05)
                )
            neighbour_frame = fe_run_scores(p, fe_scores(p, stocks))
            nr = fe_series(neighbour_frame)
            if not nr.index.equals(r.index):
                raise ValueError("Neighbour changed date coverage.")
            q = fe_quality(nr)["Backtest Quality"]
            drop = (metrics["Backtest Quality"] - q) / neighbour_sd if neighbour_sd > 0 else np.nan
            _, neighbour_traded = fe_costs(neighbour_frame)
            neighbour_net = nr.to_numpy() - FE_TRADING_FEE * neighbour_traded
            passed = bool(np.prod(1 + neighbour_net) > 1)
            neighbours.append(
                {
                    **{k: p[k] for k in FE_SETTINGS},
                    "Backtest Quality": q,
                    "Drop (sigma)": drop,
                    "Passed": passed,
                }
            )
        neighbour_df = pd.DataFrame(neighbours)
        out["Neighbourhood Cost Survival Rate"] = (
            neighbour_df["Passed"].mean() if neighbour_df["Passed"].notna().all() else np.nan
        )
        ur = fe_series(fe_run_scores(row, fe_scores(row, unseen)))
        if not ur.index.equals(r.index):
            raise ValueError(f"{sid}: unseen history differs; align source universes explicitly.")
        uq = fe_quality(ur)["Backtest Quality"]
        out["Unseen Backtest Quality"] = uq
        uscore = (
            (float(benchmark_rows.loc[sid, "Backtest Quality"]) - uq) / unseen_sd
            if unseen_sd > 0
            else np.nan
        )
        out["Unseen Gate Passed"] = (
            bool(uscore < unseen_gate) if np.isfinite(uscore) and np.isfinite(unseen_gate) else None
        )
        if np.isfinite(uscore) and pd.notna(out["Unseen Stock Score"]):
            if not np.isclose(uscore, float(out["Unseen Stock Score"]), rtol=1e-07, atol=1e-10):
                raise ValueError(f"{sid}: unseen score changed; inspect data/settings.")
        type_details = []
        for key in market_grid.columns:
            if not key.startswith("Type Score | ") or key not in row:
                continue
            a, b = (float(row[key]), float(benchmark_rows.loc[sid, key]))
            sd = market_grid[key].std()
            if np.isfinite(a) and np.isfinite(b) and (sd > 0):
                type_details.append(
                    {"Type": key[13:], "Strategy": a, "Market": b, "Relative (sigma)": (a - b) / sd}
                )
        out["Mean Relative Target Score (sigma)"] = (
            np.mean([t["Relative (sigma)"] for t in type_details]) if type_details else np.nan
        )
        out["Standard Annual Sharpe"] = fe_sharpe(r)
        out["PSR (IID)"] = fe_psr(r)
        costs, traded = fe_costs(frame)
        out.update(costs)
        out["Daily Expected Shortfall 95%"] = fe_es(r)
        details[sid] = {
            "Saved Row": row.to_dict(),
            "Daily Results": frame,
            "Daily Traded Notional": traded,
            "Stock Removals": removal_df,
            "Rolling Periods": pd.DataFrame(periods),
            "Neighbours": neighbour_df,
            "Unseen Returns": ur,
            "Target Scores": pd.DataFrame(type_details),
        }
        details[sid]["Recomputed Summary"] = out.copy()
        for destination, source in FE_EXISTING_COLUMNS.items():
            if source in row and pd.notna(row[source]):
                out[destination] = row[source]
        if "Relative Target Score" in out:
            out.pop("Mean Relative Target Score (sigma)", None)
        rows.append(out)
    unique_grid = grid.drop_duplicates(FE_SETTINGS)
    trials = len(unique_grid) if FE_DSR_TRIALS is None else int(FE_DSR_TRIALS)
    if trials < len(unique_grid):
        raise ValueError("FE_DSR_TRIALS must cover at least the full saved unique grid.")
    sample = unique_grid.sample(min(FE_DSR_SAMPLE_SIZE, len(unique_grid)), random_state=FE_SEED)
    trial_rows = []
    reference_index = next(iter(streams.values())).index
    for number, (_, row) in enumerate(sample.iterrows(), 1):
        sid = row["Simulation ID"]
        log.info("DSR calibration rerun %d/%d", number, len(sample))
        rr = (
            streams[sid]
            if sid in streams
            else fe_series(fe_run_scores(row, fe_scores(row, stocks)))
        )
        if not rr.index.equals(reference_index):
            raise ValueError("DSR calibration histories differ.")
        trial_rows.append({"Simulation ID": sid, "Daily Sharpe": fe_sharpe(rr, annual=False)})
    trial_df = pd.DataFrame(trial_rows)
    sd = trial_df["Daily Sharpe"].std()
    threshold = np.nan
    if trials == 1:
        threshold = 0.0
    elif np.isfinite(sd) and sd > 0 and trial_df["Daily Sharpe"].notna().all():
        normal = NormalDist()
        gamma = 0.5772156649015329
        threshold = sd * (
            (1 - gamma) * normal.inv_cdf(1 - 1 / trials)
            + gamma * normal.inv_cdf(1 - 1 / (trials * np.e))
        )
    result = pd.DataFrame(rows).set_index("Simulation ID")
    result["DSR (estimated, IID)"] = [fe_psr(streams[sid], threshold) for sid in result.index]
    returns = pd.concat(streams, axis=1)
    if returns.isna().any().any():
        raise ValueError("Finalist dates differ; correlations must use common coverage.")
    corr = returns.corr(min_periods=30)
    corr = corr.mask(np.eye(len(corr), dtype=bool))
    result["Average Finalist Correlation"] = corr.mean().reindex(result.index)
    result = result.reset_index()
    result.attrs.update(
        {
            "trading_fee_fraction": FE_TRADING_FEE,
            "trading_fee_percent": FE_TRADING_FEE * 100,
            "cost_bps_per_buy_plus_sell_notional": FE_TRADING_FEE * 10000,
            "annual_risk_free_rate": FE_RF_ANNUAL,
            "DSR_trial_count": trials,
            "DSR_calibration_sample_size": len(sample),
            "DSR_daily_Sharpe_dispersion": sd,
            "DSR_daily_threshold": threshold,
            "DSR_assumption": "Saved grid trials treated as independent; earlier research not inferred.",
            "PSR_DSR_assumption": "IID moment approximation; serial dependence is not corrected.",
            "turnover_convention": "Half absolute target-weight changes, initial entry included; no drift or final liquidation.",
            "cost_model": "Linear target-weight trade cost; illustrative, no liquidity/size-dependent impact.",
            "core_conventions": "Annualized mean periodic return/sample volatility; initial NAV 1 included in drawdown; scores at t earn Return at t+1.",
            "rolling_quality": "252-row windows, stride 21; original full-history benchmark denominators.",
            "neighbourhood_pass": "Fresh 30 draws; positive compounded return after configured 1x costs; original combined score preserved.",
            "unseen_pass": "One portfolio-level original gate; not a per-stock success percentage.",
            "units": "Returns, drawdowns, ES and pass rates are fractions; turnover is multiples/year.",
        }
    )
    return (result, details, returns, corr, trial_df)


def main(argv=None):
    global BACKTEST_DATABASE, SIMULATION_RESULTS_DATABASE
    global stocks, market, horizon_score_configurations
    global market_return, market_sharpe, market_max_drawdown, market_average_drawdown
    global FE_TRADING_FEE, FE_DSR_SAMPLE_SIZE, FE_DSR_TRIALS, FE_RF_ANNUAL
    global FE_NEIGHBOURHOOD_SD, FE_UNSEEN_GATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(str(data_root()) + ""))
    parser.add_argument("--output-dir", type=Path, default=get_setting("OUTPUT_DIR"))
    fees = parser.add_mutually_exclusive_group()
    fees.add_argument(
        "--trading-fee",
        type=float,
        help="Fee as a return-style fraction: 0.001 = 0.1%%; default 0.001.",
    )
    fees.add_argument("--fee-percent", type=float, help="Fee in percentage points: 0.1 = 0.1%%.")
    fees.add_argument("--cost-bps", type=float, help="Legacy basis-point input: 10 = 0.1%%.")
    parser.add_argument("--risk-free-rate", type=float, default=FE_RF_ANNUAL)
    parser.add_argument("--dsr-sample-size", type=int, default=FE_DSR_SAMPLE_SIZE)
    parser.add_argument("--dsr-trials", type=int, default=FE_DSR_TRIALS)
    parser.add_argument(
        "--neighbourhood-sd",
        type=float,
        default=FE_NEIGHBOURHOOD_SD,
        help="Optional original random-neighbour cohort SD; never estimate from finalists.",
    )
    parser.add_argument("--unseen-gate", type=float, default=FE_UNSEEN_GATE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    fee = (
        args.trading_fee
        if args.trading_fee is not None
        else args.fee_percent / 100
        if args.fee_percent is not None
        else args.cost_bps / 10000
        if args.cost_bps is not None
        else FE_TRADING_FEE
    )
    try:
        FE_TRADING_FEE = validated_fee(fee)
    except ValueError as error:
        parser.error(str(error))
    FE_RF_ANNUAL = args.risk_free_rate
    FE_DSR_SAMPLE_SIZE, FE_DSR_TRIALS = (args.dsr_sample_size, args.dsr_trials)
    FE_NEIGHBOURHOOD_SD, FE_UNSEEN_GATE = (args.neighbourhood_sd, args.unseen_gate)
    BACKTEST_DATABASE = args.data_dir / "Backtest_Database.db"
    SIMULATION_RESULTS_DATABASE = args.data_dir / "Portfolio_Simulation_Results.db"
    horizons_file = args.data_dir / "Top_Horizon_Scores.txt"
    for path in (BACKTEST_DATABASE, SIMULATION_RESULTS_DATABASE, horizons_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    horizon_score_configurations = ast.literal_eval(horizons_file.read_text())
    if not isinstance(horizon_score_configurations, list) or not all(
        (isinstance(c, dict) for c in horizon_score_configurations)
    ):
        raise TypeError("Top_Horizon_Scores.txt must contain a list of dictionaries.")
    with sqlite3.connect(BACKTEST_DATABASE.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        market = pd.read_sql_query('SELECT * FROM "Market"', connection)
        stocks = pd.read_sql_query('SELECT * FROM "Stocks"', connection)
    for frame in (market, stocks):
        frame["Date"] = pd.to_datetime(frame["Date"])
        if frame.empty:
            raise ValueError("Market and Stocks must contain data.")
    from equity_selector.metrics import performance_metrics

    benchmark = market.groupby("Date", as_index=False).agg(Return=("Return", "first"))
    benchmark_summary = performance_metrics(benchmark["Return"].dropna())
    market_return = benchmark_summary["Return"]
    market_sharpe = benchmark_summary["Sharpe Ratio"]
    market_average_drawdown = benchmark_summary["Average Drawdown"]
    market_max_drawdown = benchmark_summary["Max Drawdown"]
    evaluation, details, returns, correlations, calibration = build_final_strategy_evaluation()
    output_dir = args.output_dir or args.data_dir / "Final Evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(output_dir / "Final_Strategy_Evaluation.csv", index=False)
    evaluation.to_pickle(output_dir / "Final_Strategy_Evaluation.pkl")
    returns.to_csv(output_dir / "Final_Strategy_Daily_Returns.csv", index_label="Date")
    correlations.to_csv(output_dir / "Final_Strategy_Correlations.csv")
    calibration.to_csv(output_dir / "DSR_Calibration.csv", index=False)
    pd.to_pickle(details, output_dir / "Final_Strategy_Evaluation_Details.pkl")
    with sqlite3.connect(output_dir / "Final_Strategy_Evaluation.db") as connection:
        write_frame(
            evaluation, "Final Strategy Evaluation", connection, if_exists="replace", index=False
        )
    metadata = {
        key: None
        if isinstance(value, (float, np.floating)) and (not np.isfinite(value))
        else value.item()
        if isinstance(value, np.generic)
        else value
        for key, value in evaluation.attrs.items()
    }
    (output_dir / "Evaluation_Assumptions.json").write_text(json.dumps(metadata, indent=2))
    print(evaluation.to_string(index=False))
    print(f"\nSaved final evaluation to: {output_dir.resolve()}")
    return evaluation


if __name__ == "__main__":
    final_strategy_evaluation = main()

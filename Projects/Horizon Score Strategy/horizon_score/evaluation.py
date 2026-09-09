"""Post-backtest diagnostics; no predictive-model calls.

252 observations/year; arithmetic excess returns for Sharpe; initial wealth 1.
Turnover is half full absolute weight changes, including initial entry from cash.
Fees are charged on FULL traded notional. Input Return must be gross of these
fees. HHI uses raw weights. Cash-only effective holdings are undefined. Regime
aggregate CAGR is annualised by exposure days, not elapsed disconnected years;
drawdowns/Calmar are supplied only per continuous episode, not stitched regimes.
"""

import json
from pprint import pformat
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
from equity_selector.metrics import performance_metrics


def sharpe(returns, annualisation=252, risk_free_rate=0.0, annual=True):
    x = np.asarray(returns, dtype=float) - np.expm1(
        np.log1p(risk_free_rate) / annualisation
    )
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return (
        x.mean() / sd * (np.sqrt(annualisation) if annual else 1) if sd > 0 else np.nan
    )


def probabilistic_sharpe(returns, benchmark=0.0, annualisation=252, risk_free_rate=0.0):
    x = np.asarray(returns, dtype=float) - np.expm1(
        np.log1p(risk_free_rate) / annualisation
    )
    if len(x) < 30 or not np.isfinite(benchmark) or x.std(ddof=1) <= 0:
        return np.nan
    sr = x.mean() / x.std(ddof=1)
    centered = x - x.mean()
    m2 = np.mean(centered**2)
    skew = np.mean(centered**3) / m2**1.5
    kurt = np.mean(centered**4) / m2**2
    variance = 1 - skew * sr + (kurt - 1) * sr**2 / 4
    return (
        NormalDist().cdf((sr - benchmark) * np.sqrt(len(x) - 1) / np.sqrt(variance))
        if variance > 0
        else np.nan
    )


def expected_shortfall(returns):
    x = np.sort(np.asarray(returns, dtype=float))
    if not len(x):
        return np.nan
    mass = 0.05 * len(x)
    whole = int(np.floor(mass))
    return (x[:whole].sum() + (mass - whole) * x[whole]) / mass


def _dates(frame):
    data = frame.copy()
    if "Date" in data:
        data = data.set_index("Date")
    if not isinstance(data.index, pd.DatetimeIndex) and pd.api.types.is_numeric_dtype(
        data.index
    ):
        raise ValueError("A Date column or datetime index is required")
    data.index = (
        pd.DatetimeIndex(pd.to_datetime(data.index)).tz_localize(None).normalize()
    )
    data.index.name = "Date"
    if data.index.has_duplicates or data.index.isna().any():
        raise ValueError("Dates must be unique and nonmissing")
    return data.sort_index()


def benchmark_returns(market):
    data = market.copy()
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or "Return")
    if isinstance(data.columns, pd.MultiIndex):
        for field in ["Return", "Adj Close", "Close"]:
            matches = [c for c in data.columns if field in c]
            if matches:
                if len(matches) != 1:
                    raise ValueError("Select a single benchmark ticker")
                dates = [c for c in data.columns if "Date" in c]
                chosen = pd.DataFrame({field: data[matches[0]]})
                if dates:
                    chosen["Date"] = data[dates[0]]
                data = chosen
                break
    data = _dates(data)
    field = next((c for c in ["Return", "Adj Close", "Close"] if c in data), None)
    if field is None:
        raise ValueError("Benchmark requires Return, Adj Close or Close")
    x = pd.to_numeric(data[field], errors="raise")
    if field != "Return":
        if (x.dropna() <= 0).any():
            raise ValueError("Benchmark prices must be positive")
        x = x.pct_change(fill_method=None)
    if np.isinf(x).any() or (x.dropna() < -1).any():
        raise ValueError("Invalid benchmark returns")
    return x.rename("Market Return")


def drawdown_episodes(r):
    columns = [
        "Peak",
        "Start",
        "Trough",
        "End",
        "Recovery",
        "Recovered",
        "Max Drawdown",
        "Duration Observations",
        "Duration Days",
        "Recovery Observations",
        "Recovery Days",
    ]
    if r.empty:
        return pd.DataFrame(columns=columns)
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax().clip(lower=1) - 1
    rows, start, peak, trough = [], None, r.index[0], None
    for i, date in enumerate(r.index):
        if dd.iloc[i] < 0 and start is None:
            start, trough = i, i
        if start is not None and dd.iloc[i] < dd.iloc[trough]:
            trough = i
        recovered = dd.iloc[i] >= 0
        if start is not None and (recovered or i == len(r) - 1):
            rows.append(
                {
                    "Peak": peak,
                    "Start": r.index[start],
                    "Trough": r.index[trough],
                    "End": date,
                    "Recovery": date if recovered else pd.NaT,
                    "Recovered": recovered,
                    "Max Drawdown": dd.iloc[trough],
                    "Duration Observations": i - start + (not recovered),
                    "Duration Days": (date - peak).days,
                    "Recovery Observations": i - trough if recovered else np.nan,
                    "Recovery Days": (date - r.index[trough]).days
                    if recovered
                    else np.nan,
                }
            )
            start = None
        if recovered:
            peak = date
    return pd.DataFrame(rows, columns=columns)


def return_metrics(returns, annualisation=252, risk_free_rate=0.0):
    r = pd.Series(returns, dtype=float)
    base = performance_metrics(r, annualisation)
    n = len(r)
    if not n:
        return {
            "Observations": 0,
            **{
                k: np.nan
                for k in [
                    "Total Return",
                    "CAGR",
                    "Annualised Volatility",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Max Drawdown",
                    "Average Drawdown",
                ]
            },
        }
    growth = 1 + base["Return"]
    cagr = growth ** (annualisation / n) - 1
    excess = r - np.expm1(np.log1p(risk_free_rate) / annualisation)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    dd = (1 + r).cumprod()
    dd = dd / dd.cummax().clip(lower=1) - 1
    episodes = (
        drawdown_episodes(r)
        if isinstance(r.index, pd.DatetimeIndex)
        else pd.DataFrame()
    )
    pos, neg = r[r > 0], r[r < 0]
    return {
        "Observations": n,
        "Total Return": base["Return"],
        "CAGR": cagr,
        "Annualised Volatility": base["Volatility"],
        "Sharpe Ratio": sharpe(r, annualisation, risk_free_rate),
        "Sortino Ratio": excess.mean() / downside * np.sqrt(annualisation)
        if downside > 0
        else np.nan,
        "Calmar Ratio": cagr / abs(base["Max Drawdown"])
        if base["Max Drawdown"] < 0
        else np.nan,
        "Max Drawdown": base["Max Drawdown"],
        "Average Drawdown": base["Average Drawdown"],
        "Current Drawdown": dd.iloc[-1],
        "Longest Drawdown Duration": episodes["Duration Observations"].max()
        if not episodes.empty
        else 0,
        "Longest Drawdown Days": episodes["Duration Days"].max()
        if not episodes.empty
        else 0,
        "Longest Recovery Period": episodes["Recovery Days"].max()
        if not episodes.empty
        else 0,
        "VaR 95%": r.quantile(0.05),
        "Expected Shortfall 95%": expected_shortfall(r),
        "Positive Day Rate": len(pos) / n,
        "Negative Day Rate": len(neg) / n,
        "Average Positive Day": pos.mean(),
        "Average Negative Day": neg.mean(),
        "Profit Factor": pos.sum() / abs(neg.sum())
        if len(neg)
        else (np.inf if len(pos) else np.nan),
        "Best Day": r.max(),
        "Worst Day": r.min(),
        "Daily Skew": r.skew(),
        "Daily Excess Kurtosis": r.kurt(),
        "Probabilistic Sharpe Ratio": probabilistic_sharpe(
            r, annualisation=annualisation, risk_free_rate=risk_free_rate
        ),
    }


def portfolio_daily(weights):
    if (
        weights.shape[1] == 0
        or not np.isfinite(weights.to_numpy()).all()
        or ((weights < 0) | (weights > 1)).any().any()
    ):
        raise ValueError("Finite ticker weights in [0, 1] are required")
    changes = weights.diff()
    changes.iloc[0] = weights.iloc[0]
    previous = weights.shift(fill_value=0)
    hhi = weights.pow(2).sum(axis=1)
    traded = changes.abs().sum(axis=1)
    return pd.DataFrame(
        {
            "Holdings": (weights > 0).sum(axis=1),
            "Gross Weight": weights.sum(axis=1),
            "Largest Position": weights.max(axis=1),
            "HHI Concentration": hhi,
            "Effective Holdings": 1 / hhi.replace(0, np.nan),
            "Turnover": 0.5 * traded,
            "Traded Notional": traded,
            "Portfolio Change": traded > 1e-12,
            "Entries": ((weights > 0) & (previous == 0)).sum(axis=1),
            "Exits": ((weights == 0) & (previous > 0)).sum(axis=1),
            "Absolute Weight Change": changes.abs().mean(axis=1),
        }
    )


def portfolio_metrics(daily, annualisation=252):
    result = {}
    for source, labels in {
        "Holdings": ["Average", "Median", "Minimum", "Maximum"],
        "Gross Weight": ["Average", "Minimum", "Maximum"],
        "Largest Position": ["Average"],
        "HHI Concentration": ["Average", "Median"],
        "Effective Holdings": ["Average", "Median", "Minimum"],
        "Turnover": ["Average", "Median", "Maximum"],
    }.items():
        for label in labels:
            fn = {
                "Average": "mean",
                "Median": "median",
                "Minimum": "min",
                "Maximum": "max",
            }[label]
            name = "Daily Turnover" if source == "Turnover" else source
            result[label + " " + name] = getattr(daily[source], fn)()
    result.update(
        {
            "Maximum Position Observed": daily["Largest Position"].max(),
            "95th Percentile Daily Turnover": daily.Turnover.quantile(0.95),
            "Annualised Turnover": daily.Turnover.mean() * annualisation,
            "Fraction of Days with Portfolio Changes": daily["Portfolio Change"].mean(),
            "Average Entries Per Day": daily.Entries.mean(),
            "Average Exits Per Day": daily.Exits.mean(),
            "Maximum Entries In One Day": daily.Entries.max(),
            "Maximum Exits In One Day": daily.Exits.max(),
            "Average Absolute Weight Change": daily["Absolute Weight Change"].mean(),
        }
    )
    return result


def cost_metrics(r, traded, trading_cost=0.001, annualisation=252, risk_free_rate=0.0):
    if not np.isfinite(trading_cost) or trading_cost < 0:
        raise ValueError("Trading cost must be finite and nonnegative")
    r, traded = np.asarray(r, dtype=float), np.asarray(traded, dtype=float)
    if np.any(1 + r - 2 * trading_cost * traded <= 0):
        raise ValueError("Cost stress causes insolvency")
    net = r - trading_cost * traded
    gross_metrics = return_metrics(r, annualisation, risk_free_rate)
    net_metrics = return_metrics(net, annualisation, risk_free_rate)
    if gross_metrics["Total Return"] <= 0:
        breakeven = 0.0
    elif not (traded > 0).any():
        breakeven = np.inf
    else:
        low, high = 0.0, np.min((1 + r[traded > 0]) / traded[traded > 0])
        for _ in range(80):
            mid = (low + high) / 2
            if np.log1p(r - mid * traded).sum() > 0:
                low = mid
            else:
                high = mid
        breakeven = (low + high) / 2 * 10000
    return {
        "Trading Cost Fraction": trading_cost,
        "Trading Cost bps": trading_cost * 10000,
        "Gross Total Return": gross_metrics["Total Return"],
        "Net Total Return": net_metrics["Total Return"],
        "Net CAGR": net_metrics["CAGR"],
        "Gross Sharpe": gross_metrics["Sharpe Ratio"],
        "Sharpe After Costs": net_metrics["Sharpe Ratio"],
        "Sharpe at 2x Costs": sharpe(
            r - 2 * trading_cost * traded, annualisation, risk_free_rate
        ),
        "Net Max Drawdown": net_metrics["Max Drawdown"],
        "Break-Even Cost bps": breakeven,
    }


def benchmark_metrics(strategy, benchmark, annualisation=252, risk_free_rate=0.0):
    aligned = pd.concat(
        [strategy.rename("Strategy"), benchmark.rename("Market")], axis=1
    ).dropna()
    s, m = aligned.Strategy, aligned.Market
    result = {"Aligned Observations": len(aligned)}
    for label, r in [("Strategy", s), ("Market", m)]:
        metrics = return_metrics(r, annualisation, risk_free_rate)
        for metric in ["Total Return", "CAGR", "Annualised Volatility", "Max Drawdown"]:
            result[label + " " + metric] = metrics[metric]
        result[label + " Sharpe"] = metrics["Sharpe Ratio"]
    variance = m.var(ddof=1)
    beta = s.cov(m) / variance if len(s) > 1 and variance > 0 else np.nan
    active = s - m
    te = active.std(ddof=1) * np.sqrt(annualisation)
    rf = np.expm1(np.log1p(risk_free_rate) / annualisation)
    result.update(
        {
            "Strategy-Market Correlation": s.corr(m)
            if len(s) > 1 and s.std() > 0 and m.std() > 0
            else np.nan,
            "Beta": beta,
            "Annualised Alpha": ((s.mean() - rf) - beta * (m.mean() - rf))
            * annualisation,
            "Tracking Error": te,
            "Information Ratio": active.mean() * annualisation / te
            if te > 0
            else np.nan,
            "Annualised Active Return": active.mean() * annualisation,
            "Upside Capture": s[m > 0].mean() / m[m > 0].mean()
            if (m > 0).any()
            else np.nan,
            "Downside Capture": s[m < 0].mean() / m[m < 0].mean()
            if (m < 0).any()
            else np.nan,
            "Strategy mean return on positive-market days": s[m > 0].mean(),
            "Strategy mean return on negative-market days": s[m < 0].mean(),
        }
    )
    return result


def evaluate_backtest(
    backtest_results,
    regimes,
    market,
    *,
    trading_cost=0.001,
    annualisation=252,
    risk_free_rate=0.0,
    windows=(63, 126, 252),
    output_dir=None,
):
    if (
        not np.isfinite(annualisation)
        or annualisation <= 0
        or not np.isfinite(risk_free_rate)
        or risk_free_rate <= -1
    ):
        raise ValueError("Invalid annualisation or risk free rate")
    data = _dates(backtest_results)
    if data.empty or not data.columns.is_unique or "Return" not in data:
        raise ValueError("Nonempty daily returns with unique columns required")
    r = pd.to_numeric(data.Return, errors="raise")
    if not np.isfinite(r).all() or (r < -1).any():
        raise ValueError("Returns must be finite and at least -1")
    pdaily = portfolio_daily(data.drop(columns="Return").astype(float))
    market_r = benchmark_returns(market)
    daily = data.join(pdaily.add_prefix("Portfolio ")).join(market_r)
    daily["Net Return"] = r - trading_cost * pdaily["Traded Notional"]
    daily["Wealth"] = (1 + r).cumprod()
    daily["Drawdown"] = daily.Wealth / daily.Wealth.cummax().clip(lower=1) - 1
    summary = return_metrics(r, annualisation, risk_free_rate)
    costs = cost_metrics(
        r, pdaily["Traded Notional"], trading_cost, annualisation, risk_free_rate
    )
    regime_rows, episode_rows = [], []
    for family, states in regimes.items():
        labels = pd.Series("Unknown", index=data.index)
        used = pd.Series(False, index=data.index)
        all_intervals = []
        for state, intervals in states.items():
            for start, end in intervals:
                start, end = pd.Timestamp(start), pd.Timestamp(end)
                if pd.isna(start) or pd.isna(end) or start > end:
                    raise ValueError("Invalid regime interval")
                all_intervals.append((start, end, state))
        all_intervals.sort()
        if any(b[0] <= a[1] for a, b in zip(all_intervals, all_intervals[1:])):
            raise ValueError(f"Overlapping {family} regime intervals")
        for start, end, state in all_intervals:
            mask = (data.index >= start) & (data.index <= end)
            labels.loc[mask] = state
            used.loc[mask] = True
            if mask.any():
                episode_rows.append(
                    {
                        "Regime Family": family,
                        "Regime": state,
                        "Start": start,
                        "End": end,
                        **return_metrics(r.loc[mask], annualisation, risk_free_rate),
                        **portfolio_metrics(pdaily.loc[mask], annualisation),
                        **benchmark_metrics(
                            r.loc[mask],
                            market_r.reindex(r.index[mask]),
                            annualisation,
                            risk_free_rate,
                        ),
                    }
                )
        daily["Regime " + family] = labels
        for state in dict.fromkeys([*states, "Unknown"]):
            mask = labels.eq(state)
            part = r.loc[mask]
            metrics = return_metrics(part, annualisation, risk_free_rate)
            bm = benchmark_metrics(
                part, market_r.reindex(part.index), annualisation, risk_free_rate
            )
            # These would describe fictitious stitched paths across episodes.
            for key in list(metrics):
                if "Drawdown" in key or "Recovery" in key or key == "Calmar Ratio":
                    metrics[key] = np.nan
            for key in ["Market Max Drawdown", "Strategy Max Drawdown"]:
                bm[key] = np.nan
            regime_rows.append(
                {
                    "Regime Family": family,
                    "Regime": state,
                    "Days": len(part),
                    "Fraction of Backtest": len(part) / len(r),
                    "Annualisation Basis": "exposure observations",
                    **metrics,
                    **portfolio_metrics(pdaily.loc[mask], annualisation),
                    **bm,
                }
            )
    rolling = pd.DataFrame(index=data.index)
    for window in windows:
        if isinstance(window, bool) or int(window) != window or window < 2:
            raise ValueError("Rolling windows must be integers >= 2")
        roll = r.rolling(window)
        total = roll.apply(lambda x: np.prod(1 + x) - 1, raw=True)
        rolling[f"{window} Rolling Return"] = total
        rolling[f"{window} Rolling Annualised Return"] = (1 + total) ** (
            annualisation / window
        ) - 1
        rolling[f"{window} Rolling Volatility"] = roll.std() * np.sqrt(annualisation)
        rolling[f"{window} Rolling Sharpe"] = roll.apply(
            lambda x: sharpe(x, annualisation, risk_free_rate), raw=True
        )
        aligned_market = market_r.reindex(r.index)
        rolling[f"{window} Rolling Market Correlation"] = roll.corr(aligned_market)
        rolling[f"{window} Rolling Beta"] = roll.cov(
            aligned_market
        ) / aligned_market.rolling(window).var().replace(0, np.nan)
        for source, name in [
            ("Holdings", "Average Holdings"),
            ("Turnover", "Average Turnover"),
            ("Effective Holdings", "Effective Holdings"),
        ]:
            rolling[f"{window} Rolling {name}"] = pdaily[source].rolling(window).mean()
    yearly = []
    for year, part in r.groupby(r.index.year):
        yearly.append(
            {
                "Year": year,
                **return_metrics(part, annualisation, risk_free_rate),
                **benchmark_metrics(
                    part, market_r.reindex(part.index), annualisation, risk_free_rate
                ),
            }
        )
    monthly = r.groupby([r.index.year, r.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly.index.names = ["Year", "Month"]
    result = {
        "summary": pd.Series(summary),
        "benchmark": pd.Series(
            benchmark_metrics(
                r, market_r.reindex(r.index), annualisation, risk_free_rate
            )
        ),
        "portfolio": pd.Series(portfolio_metrics(pdaily, annualisation)),
        "costs": pd.Series(costs),
        "regimes": pd.DataFrame(regime_rows),
        "regime_episodes": pd.DataFrame(episode_rows),
        "rolling": rolling,
        "yearly": pd.DataFrame(yearly)
        .set_index("Year")
        .rename(
            columns={
                "Strategy Total Return": "Strategy Return",
                "Market Total Return": "Market Return",
            }
        ),
        "monthly": monthly.unstack().reindex(columns=range(1, 13)),
        "drawdowns": drawdown_episodes(r),
        "daily": daily,
    }
    if output_dir is not None:
        save_evaluation(result, output_dir)
        output = Path(output_dir)
        frozen = {
            family: {
                state: [
                    (str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
                    for start, end in intervals
                ]
                for state, intervals in states.items()
            }
            for family, states in regimes.items()
        }
        (output / "Macro_Regimes.txt").write_text(
            pformat(frozen, sort_dicts=False) + "\n"
        )
        (output / "Assumptions.json").write_text(
            json.dumps(
                {
                    "annualisation": annualisation,
                    "risk_free_rate": risk_free_rate,
                    "trading_cost_fraction": trading_cost,
                    "rolling_windows": list(windows),
                    "input_returns": "gross",
                    "turnover": "half absolute weight changes including initial entry",
                    "fees": "full absolute weight changes",
                    "regime_annualisation": "exposure observations",
                },
                indent=2,
            )
            + "\n"
        )
    return result


def save_evaluation(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        value.to_csv(output / (name + ".csv"))
    (output / "Conventions.txt").write_text(
        __doc__
        + "\nVaR and ES are signed returns; capture ratios use conditional arithmetic means.\nRisk-free rate defaults to zero; PSR assumes IID returns and needs 30 observations.\nUnrecovered drawdowns have no recovery duration. Initial entry counts as trading.\n"
    )

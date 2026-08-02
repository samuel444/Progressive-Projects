
"""Presentation-focused charts used by the project notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(fig: plt.Figure, save_path: Optional[Path] = None) -> plt.Figure:
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def plot_price_history(close_data: pd.DataFrame, save_path: Optional[Path] = None):
    ax = close_data["Close"].plot(figsize=(12, 6), title="Adjusted Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    return _finish(ax.figure, save_path)


def plot_realised_volatility(feature_data: pd.DataFrame, ticker: str, save_path=None):
    columns = [(name, ticker) for name in ("RV20", "RV60", "RV252")]
    ax = feature_data[columns].droplevel(1, axis=1).plot(
        figsize=(12, 6), title=f"{ticker} Annualised Realised Volatility"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised volatility")
    return _finish(ax.figure, save_path)


def plot_market_vs_model(options: pd.DataFrame, save_path=None):
    valid = options[["MarketMid", "BS_ForeV"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(valid["MarketMid"], valid["BS_ForeV"], alpha=0.6)
    if not valid.empty:
        lower = min(valid.min())
        upper = max(valid.max())
        ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel("Market midpoint")
    ax.set_ylabel("Black-Scholes forecast-vol price")
    ax.set_title("Market Price vs Model Price")
    return _finish(fig, save_path)


def plot_portfolio_greeks(portfolio_risk: pd.DataFrame, save_path=None):
    data = portfolio_risk.loc[portfolio_risk["Ticker"].ne("PORTFOLIO")].set_index("Ticker")
    greeks = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
    fig, axes = plt.subplots(len(greeks), 1, figsize=(11, 15))
    for ax, greek in zip(axes, greeks):
        data[greek].plot(kind="bar", ax=ax, title=f"Position {greek} by Ticker")
        ax.set_xlabel("")
        ax.set_ylabel(greek)
    return _finish(fig, save_path)


def plot_scenario_pnl_distribution(scenario_portfolio: pd.DataFrame, save_path=None):
    pnl = scenario_portfolio.loc[
        scenario_portfolio["Scenario ID"].ne("BASE"), "Portfolio_PnL"
    ].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pnl, bins=50)
    ax.axvline(0, linestyle="--", label="Zero P&L")
    if not pnl.empty:
        ax.axvline(pnl.quantile(0.05), linestyle=":", label="5th percentile")
    ax.set_xlabel("Portfolio scenario P&L")
    ax.set_ylabel("Frequency")
    ax.set_title("Portfolio Scenario P&L Distribution")
    ax.legend()
    return _finish(fig, save_path)


def plot_worst_ticker_contributions(contributions: pd.DataFrame, save_path=None):
    data = contributions.sort_values("Scenario_PnL")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data["Ticker"], data["Scenario_PnL"])
    ax.set_xlabel("Scenario P&L")
    ax.set_title("Ticker Contributions in the Worst Scenario")
    return _finish(fig, save_path)


def plot_full_vs_approximate_pnl(scenario_attribution: pd.DataFrame, save_path=None):
    data = scenario_attribution.loc[
        scenario_attribution["Scenario ID"].ne("BASE")
    ].dropna(subset=["Full_Revaluation_PnL", "Approximate_PnL"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(data["Approximate_PnL"], data["Full_Revaluation_PnL"], alpha=0.5)
    if not data.empty:
        low = min(data["Approximate_PnL"].min(), data["Full_Revaluation_PnL"].min())
        high = max(data["Approximate_PnL"].max(), data["Full_Revaluation_PnL"].max())
        ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Greek approximation P&L")
    ax.set_ylabel("Full-revaluation P&L")
    ax.set_title("Greek Approximation vs Full Revaluation")
    return _finish(fig, save_path)


def plot_attribution_components(row: pd.Series, save_path=None):
    components = pd.Series({
        "Delta": row["Delta_PnL"],
        "Gamma": row["Gamma_PnL"],
        "Vega": row["Vega_PnL"],
        "Theta": row["Theta_PnL"],
        "Rho": row["Rho_PnL"],
        "Residual": row["Residual_PnL"],
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    components.plot(kind="bar", ax=ax)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("P&L contribution")
    ax.set_title(f"Greek P&L Attribution — {row['Scenario ID']}")
    return _finish(fig, save_path)


def plot_residual_vs_spot_shock(scenario_attribution: pd.DataFrame, save_path=None):
    data = scenario_attribution.loc[scenario_attribution["Scenario ID"].ne("BASE")]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(data["Spot Shock"].abs(), data["Gross Residual %"], alpha=0.5)
    ax.set_xlabel("Absolute spot shock")
    ax.set_ylabel("Gross residual percentage")
    ax.set_title("Attribution Error vs Scenario Size")
    return _finish(fig, save_path)

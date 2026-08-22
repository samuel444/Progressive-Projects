
"""Monte Carlo option-chain validation and terminal profit distributions."""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from options_risk_engine.config import MONTE_CARLO_SIMULATIONS, PLOT_MONTE_CARLO
from options_risk_engine.domain import OptionTicker
from options_risk_engine.utils import safe_relative_edge

logger = logging.getLogger(__name__)

def monte_carlo_option_chain(
    ticker: OptionTicker,
    simulations: int = 10000,
    random_seed: Optional[int] = None,
):
    """Price one ticker's chains using the same stored market assumptions."""

    ticker.ensure_pricing_ready()

    if "ForeV" not in ticker.volatility_inputs:
        raise ValueError(
            f"{ticker.symbol} does not have a ForeV volatility input"
        )

    logger.info(
        "%s starting Monte Carlo simulation",
        ticker.symbol,
    )

    call_chain = ticker.call_chain.copy()
    put_chain = ticker.put_chain.copy()

    time_to_expiry = ticker.time_to_expiry
    steps = max(int(ticker.trading_dte), 1)
    dt = time_to_expiry / steps

    current_price = ticker.current_price
    sigma = ticker.volatility_inputs["ForeV"]
    risk_free_rate = ticker.risk_free_rate
    dividend_yield = ticker.dividend_yield

    logger.info(
        "%s Monte Carlo inputs - S0: %.2f, sigma: %.4f, "
        "T: %.4f, steps: %d, simulations: %d",
        ticker.symbol,
        current_price,
        sigma,
        time_to_expiry,
        steps,
        simulations,
    )

    rng = np.random.default_rng(random_seed)

    half_simulations = (simulations + 1) // 2
    z_half = rng.standard_normal(
        size=(steps, half_simulations)
    )
    z = np.concatenate([z_half, -z_half], axis=1)
    z = z[:, :simulations]

    log_increments = (
        (
            risk_free_rate
            - dividend_yield
            - 0.5 * sigma**2
        )
        * dt
        + sigma * np.sqrt(dt) * z
    )

    cumulative_log_returns = np.cumsum(
        log_increments,
        axis=0,
    )
    cumulative_log_returns = np.vstack(
        [
            np.zeros(simulations),
            cumulative_log_returns,
        ]
    )

    paths = current_price * np.exp(cumulative_log_returns)
    terminal_prices = paths[-1]
    discount_factor = np.exp(
        -risk_free_rate * time_to_expiry
    )

    call_strikes = call_chain["strike"].to_numpy(dtype=float)
    call_payoffs = np.maximum(
        terminal_prices[:, None] - call_strikes[None, :],
        0.0,
    )
    call_prices = discount_factor * call_payoffs.mean(axis=0)
    call_standard_errors = (
        discount_factor
        * call_payoffs.std(axis=0, ddof=1)
        / np.sqrt(simulations)
    )

    put_strikes = put_chain["strike"].to_numpy(dtype=float)
    put_payoffs = np.maximum(
        put_strikes[None, :] - terminal_prices[:, None],
        0.0,
    )
    put_prices = discount_factor * put_payoffs.mean(axis=0)
    put_standard_errors = (
        discount_factor
        * put_payoffs.std(axis=0, ddof=1)
        / np.sqrt(simulations)
    )

    call_chain["MC_ForeV"] = call_prices
    call_chain["MC_SE"] = call_standard_errors
    call_chain["MC AskEdge"] = safe_relative_edge(
        call_chain["MC_ForeV"],
        call_chain["ask"],
    )

    put_chain["MC_ForeV"] = put_prices
    put_chain["MC_SE"] = put_standard_errors
    put_chain["MC AskEdge"] = safe_relative_edge(
        put_chain["MC_ForeV"],
        put_chain["ask"],
    )

    logger.info(
        "%s Monte Carlo simulation complete",
        ticker.symbol,
    )

    result = {
        "paths": paths,
        "terminal_prices": terminal_prices,
        "calls": call_chain,
        "puts": put_chain,
        "sigma": sigma,
        "T": time_to_expiry,
        "steps": steps,
    }
    ticker.monte_carlo_result = result
    return result


def plot_monte_carlo_result(
    ticker: OptionTicker,
    paths: np.ndarray,
    terminal_prices: np.ndarray,
    path_count: int = 100,
) -> None:
    """Plot a sample of paths and the terminal-price distribution."""

    median_path = np.median(paths, axis=1)
    shown_paths = min(path_count, paths.shape[1])

    '''plt.figure()
    plt.plot(paths[:, :shown_paths], alpha=0.4)
    plt.plot(
        median_path,
        linewidth=3,
        label="Median simulation",
    )
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price ($)")
    plt.title(f"{ticker.symbol} Monte Carlo Simulation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(terminal_prices, bins=50)
    plt.axvline(
        ticker.current_price,
        linestyle="--",
        label="Current Price",
    )
    plt.xlabel("Price at Expiry ($)")
    plt.ylabel("Frequency")
    plt.title(f"{ticker.symbol} Price Distribution at Expiry")
    plt.legend()
    plt.tight_layout()
    plt.show()'''


def add_option_profit_distribution(
    options: pd.DataFrame,
    terminal_prices: np.ndarray,
    option_type: str,
    risk_free_rate: float,
    time_to_expiry: float,
    include_premium_carry: bool = True,
) -> pd.DataFrame:
    """Calculate vectorised terminal-profit metrics for every contract.

    The output is per share. Multiply by 100 for a standard US equity option
    contract. When include_premium_carry is True, the premium is grown at the
    risk-free rate to put both the premium and payoff on the expiry-date basis.
    """

    result = options.copy()

    premium = result["ask"].fillna(result["BS_ForeV"])

    result["Premium Used"] = premium
    result["Premium Source"] = np.where(
        result["ask"].notna(),
        "Ask",
        "BS_ForeV",
    )

    premium_multiplier = (
        np.exp(risk_free_rate * time_to_expiry)
        if include_premium_carry
        else 1.0
    )

    premium_at_expiry = premium.to_numpy(dtype=float) * premium_multiplier
    strikes = result["strike"].to_numpy(dtype=float)
    terminal_prices = np.asarray(terminal_prices, dtype=float)

    if option_type.lower() == "call":
        payoffs = np.maximum(
            terminal_prices[:, None] - strikes[None, :],
            0.0,
        )
    elif option_type.lower() == "put":
        payoffs = np.maximum(
            strikes[None, :] - terminal_prices[:, None],
            0.0,
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    profits = payoffs - premium_at_expiry[None, :]

    percentiles = np.percentile(
        profits,
        [5, 50, 95],
        axis=0,
    )

    result["Premium at Expiry"] = premium_at_expiry
    result["Lower Profit"] = percentiles[0]
    result["Median Profit"] = percentiles[1]
    result["Upper Profit"] = percentiles[2]
    result["Expected Profit"] = profits.mean(axis=0)
    result["Probability of Profit"] = (profits > 0).mean(axis=0)

    # Returns normalise contracts with very different premiums and strikes.
    valid_premium = premium_at_expiry > 0

    for profit_column, return_column in (
        ("Lower Profit", "Lower Profit Return"),
        ("Median Profit", "Median Profit Return"),
        ("Upper Profit", "Upper Profit Return"),
        ("Expected Profit", "Expected Profit Return"),
    ):
        result[return_column] = np.nan
        result.loc[valid_premium, return_column] = (
            result.loc[valid_premium, profit_column]
            / premium_at_expiry[valid_premium]
        )

    return result


def run_monte_carlo_analysis(
    tickers: list[OptionTicker],
    cleaned_options: dict[str, pd.DataFrame],
    simulations: int = MONTE_CARLO_SIMULATIONS,
    plot_results: bool = PLOT_MONTE_CARLO,
    random_seed: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    """Run Monte Carlo analysis using each ticker object's stored state."""

    monte_carlo_results: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        call_key = ticker.table_key("Calls")
        put_key = ticker.table_key("Puts")

        if (
            ticker.call_chain is None
            or ticker.put_chain is None
            or call_key not in cleaned_options
            or put_key not in cleaned_options
        ):
            logger.warning(
                "%s skipped in Monte Carlo because required data is unavailable",
                ticker.symbol,
            )
            continue

        logger.info(
            "Running %d Monte Carlo simulations for %s",
            simulations,
            ticker.symbol,
        )

        try:
            result = monte_carlo_option_chain(
                ticker=ticker,
                simulations=simulations,
                random_seed=random_seed,
            )
            paths = np.asarray(result["paths"], dtype=float)
            terminal_prices = np.asarray(
                result["terminal_prices"],
                dtype=float,
            )

            if terminal_prices.ndim != 1 or terminal_prices.size == 0:
                raise ValueError(
                    f"Invalid terminal-price array for {ticker.symbol}: "
                    f"shape {terminal_prices.shape}"
                )

            if plot_results:
                plot_monte_carlo_result(
                    ticker=ticker,
                    paths=paths,
                    terminal_prices=terminal_prices,
                )

            for option_key, result_key, option_type in (
                (call_key, "calls", "call"),
                (put_key, "puts", "put"),
            ):
                mc_columns = result[result_key][[
                    "contractSymbol",
                    "MC_ForeV",
                    "MC_SE",
                    "MC AskEdge",
                ]]

                options = cleaned_options[option_key].merge(
                    mc_columns,
                    on="contractSymbol",
                    how="left",
                    validate="one_to_one",
                )
                options["MC - BS ForeV"] = (
                    options["MC_ForeV"]
                    - options["BS_ForeV"]
                )
                options["MC Difference in SE"] = np.where(
                    options["MC_SE"] > 0,
                    options["MC - BS ForeV"] / options["MC_SE"],
                    np.nan,
                )

                cleaned_options[option_key] = add_option_profit_distribution(
                    options=options,
                    terminal_prices=terminal_prices,
                    option_type=option_type,
                    risk_free_rate=ticker.risk_free_rate,
                    time_to_expiry=ticker.time_to_expiry,
                )

            ticker.call_analysis = cleaned_options[call_key]
            ticker.put_analysis = cleaned_options[put_key]
            ticker.monte_carlo_result = result
            monte_carlo_results[ticker.symbol] = result

            logger.info(
                "%s Monte Carlo profit analysis complete",
                ticker.symbol,
            )

        except Exception:
            logger.exception(
                "Monte Carlo analysis failed for %s",
                ticker.symbol,
            )

    return monte_carlo_results

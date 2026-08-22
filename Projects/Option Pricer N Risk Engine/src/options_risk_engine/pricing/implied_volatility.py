
"""Market implied-volatility inversion and quote-level checks."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from options_risk_engine.pricing.black_scholes import objective_function, option_price_bounds

def implied_volatility(
    ticker,
    strike,
    market_price,
    option_type,
    lower_volatility=1e-6,
    upper_volatility=5.0,
):
    """Calculate implied volatility using Brent's method."""

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    if (
        not np.isfinite(market_price)
        or market_price <= 0
        or not np.isfinite(strike)
        or strike <= 0
        or ticker.time_to_expiry <= 0
    ):
        return np.nan

    lower_price_bound, upper_price_bound = (
        option_price_bounds(
            ticker=ticker,
            strike=strike,
            option_type=option_type,
        )
    )

    price_tolerance = 1e-8

    if (
        market_price < lower_price_bound - price_tolerance
        or market_price > upper_price_bound + price_tolerance
    ):
        return np.nan

    objective_args = (
        ticker,
        market_price,
        strike,
        ticker.time_to_expiry,
        ticker.risk_free_rate,
        ticker.dividend_yield,
        option_type,
    )

    lower_objective = objective_function(
        lower_volatility,
        *objective_args,
    )

    upper_objective = objective_function(
        upper_volatility,
        *objective_args,
    )

    if (
        not np.isfinite(lower_objective)
        or not np.isfinite(upper_objective)
        or lower_objective * upper_objective > 0
    ):
        return np.nan

    try:
        return float(
            brentq(
                objective_function,
                lower_volatility,
                upper_volatility,
                args=objective_args,
                xtol=1e-6,
                rtol=1e-6,
                maxiter=1000,
            )
        )

    except (ValueError, RuntimeError):
        return np.nan


def add_implied_volatility_columns(
    options: pd.DataFrame,
    ticker,
    option_type: str,
) -> pd.DataFrame:
    """
    Calculate implied volatility from the bid, midpoint and ask
    for every option in one call or put table.
    """

    result = options.copy()

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    market_price_columns = {
        "bid": "IV_Bid",
        "MarketMid": "IV_Mid",
        "ask": "IV_Ask",
    }

    for market_column, iv_column in market_price_columns.items():

        result[iv_column] = [
            implied_volatility(
                ticker=ticker,
                strike=float(strike),
                market_price=float(market_price)
                if pd.notna(market_price)
                else np.nan,
                option_type=option_type,
            )
            for strike, market_price in zip(
                result["strike"],
                result[market_column],
            )
        ]

    # Width of the market's implied-volatility spread
    result["IV Bid-Ask Spread"] = (
        result["IV_Ask"]
        - result["IV_Bid"]
    )

    # Compare your calculated midpoint IV with Yahoo's IV
    result["IV Mid - Yahoo IV"] = (
        result["IV_Mid"]
        - result["IV Used"]
    )

    # Check the expected IV ordering
    result["IV Order Valid"] = (
        result["IV_Bid"].notna()
        & result["IV_Mid"].notna()
        & result["IV_Ask"].notna()
        & result["IV_Bid"].le(result["IV_Mid"] + 1e-8)
        & result["IV_Mid"].le(result["IV_Ask"] + 1e-8)
    )

    return result

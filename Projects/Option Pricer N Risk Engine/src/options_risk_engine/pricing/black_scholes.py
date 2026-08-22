
"""Black-Scholes pricing and scalar helper functions."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def black_scholes(
    ticker,
    spot,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield,
    option_type,
    volatility=None,
):
    """Price European calls or puts for scalar or array-like inputs.

    This keeps the original public signature but resolves volatility before the
    d1/d2 calculation and handles expiry explicitly.  Those two changes make
    the function safe for the scenario engine when shocked time reaches zero.
    """
    option_type = str(option_type).lower().rstrip("s")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if volatility is None:
        if option_type == "call":
            volatility = ticker.call_chain["impliedVolatility"]
        else:
            volatility = ticker.put_chain["impliedVolatility"]

    spot_a, strike_a, time_a, rate_a, yield_a, vol_a = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(time_to_expiry, dtype=float),
        np.asarray(risk_free_rate, dtype=float),
        np.asarray(dividend_yield, dtype=float),
        np.asarray(volatility, dtype=float),
    )

    result = np.full(spot_a.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(spot_a) & np.isfinite(strike_a) & np.isfinite(time_a)
        & np.isfinite(rate_a) & np.isfinite(yield_a) & np.isfinite(vol_a)
        & (spot_a > 0) & (strike_a > 0) & (time_a >= 0) & (vol_a > 0)
    )

    expired = valid & (time_a <= 0)
    if option_type == "call":
        result[expired] = np.maximum(spot_a[expired] - strike_a[expired], 0.0)
    else:
        result[expired] = np.maximum(strike_a[expired] - spot_a[expired], 0.0)

    live = valid & (time_a > 0)
    if live.any():
        sqrt_t = np.sqrt(time_a[live])
        d1 = (
            np.log(spot_a[live] / strike_a[live])
            + (rate_a[live] - yield_a[live] + 0.5 * vol_a[live] ** 2)
            * time_a[live]
        ) / (vol_a[live] * sqrt_t)
        d2 = d1 - vol_a[live] * sqrt_t

        if option_type == "call":
            result[live] = (
                spot_a[live] * np.exp(-yield_a[live] * time_a[live]) * norm.cdf(d1)
                - strike_a[live] * np.exp(-rate_a[live] * time_a[live]) * norm.cdf(d2)
            )
        else:
            result[live] = (
                strike_a[live] * np.exp(-rate_a[live] * time_a[live]) * norm.cdf(-d2)
                - spot_a[live] * np.exp(-yield_a[live] * time_a[live]) * norm.cdf(-d1)
            )

    return float(result) if result.ndim == 0 else result


def years_to_expiry(
    expiry: str,
    valuation_date: pd.Timestamp,
) -> float:
    """Return calendar time to expiry in years using an ACT/365 convention."""

    if valuation_date is None:
        valuation_date = pd.Timestamp.today().normalize()
    else:
        valuation_date = pd.Timestamp(valuation_date).normalize()

    expiry_date = pd.Timestamp(expiry).normalize()
    calendar_days = max((expiry_date - valuation_date).days, 0)

    return calendar_days / 365.0


def option_price_bounds(
    ticker,
    strike,
    option_type,
):
    spot = ticker.current_price
    time_to_expiry = ticker.time_to_expiry
    risk_free_rate = ticker.risk_free_rate
    dividend_yield = ticker.dividend_yield

    discounted_spot = (
        spot
        * np.exp(-dividend_yield * time_to_expiry)
    )

    discounted_strike = (
        strike
        * np.exp(-risk_free_rate * time_to_expiry)
    )

    if option_type == "call":
        lower_bound = max(
            discounted_spot - discounted_strike,
            0.0,
        )

        upper_bound = discounted_spot

    elif option_type == "put":
        lower_bound = max(
            discounted_strike - discounted_spot,
            0.0,
        )

        upper_bound = discounted_strike

    else:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    return lower_bound, upper_bound


def objective_function(volatility, 
                ticker,
                market_price,
                strike,
                time_to_expiry,
                risk_free_rate,
                dividend_yield,
                option_type
):
        return (
            black_scholes(
                ticker,
                ticker.current_price,
                strike,
                time_to_expiry,
                risk_free_rate,
                dividend_yield,
                option_type,
                volatility
            )
            - market_price
        )


def scalar_black_scholes_price(
    ticker,
    spot,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield,
    option_type,
    volatility,
):
    """Return one Black-Scholes price as a scalar float."""

    value = black_scholes(
        ticker=ticker,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
        volatility=volatility,
    )

    array = np.asarray(
        value,
        dtype=float,
    )

    if array.size != 1:
        raise ValueError(
            "Numerical Greek validation requires a scalar option price"
        )

    return float(
        array.reshape(-1)[0]
    )

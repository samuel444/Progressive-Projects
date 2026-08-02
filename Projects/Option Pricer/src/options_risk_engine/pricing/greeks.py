
"""Analytical Greeks, sign checks and finite-difference validation."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from options_risk_engine.config import GREEK_VALIDATION_TOLERANCES
from options_risk_engine.pricing.black_scholes import scalar_black_scholes_price

logger = logging.getLogger(__name__)

def calendar_days_to_trading_days(
    calendar_days: int,
    valuation_date: Optional[pd.Timestamp] = None,
) -> int:
    """Convert calendar days into weekdays from one valuation date."""

    if calendar_days < 0:
        raise ValueError("calendar_days cannot be negative")

    if valuation_date is None:
        valuation_date = pd.Timestamp.today().normalize()
    else:
        valuation_date = pd.Timestamp(valuation_date).normalize()

    target_date = valuation_date + pd.Timedelta(days=calendar_days)
    trading_days = int(
        np.busday_count(
            valuation_date.date(),
            target_date.date(),
        )
    )

    logger.info(
        "Converted %d calendar days from %s into %d approximate trading days",
        calendar_days,
        valuation_date.date(),
        trading_days,
    )

    return trading_days


def black_scholes_greeks(
    ticker,
    strike,
    volatility,
    option_type,
    time_to_expiry=None,
):
    """
    Calculate Black-Scholes Greeks for one European option.

    Units
    -----
    Delta:
        Option-price change per $1 stock-price change.

    Gamma:
        Delta change per $1 stock-price change.

    Vega:
        Option-price change per one volatility percentage point.

    Theta:
        Option-price change per one calendar day passing.

    Rho:
        Option-price change per one interest-rate percentage point.
    """

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    spot = float(ticker.current_price)
    strike = float(strike)
    volatility = float(volatility)

    if time_to_expiry is None:
        time_to_expiry = float(ticker.time_to_expiry)
    else:
        time_to_expiry = float(time_to_expiry)

    risk_free_rate = float(ticker.risk_free_rate)
    dividend_yield = float(ticker.dividend_yield)

    if (
        not np.isfinite(spot)
        or spot <= 0
        or not np.isfinite(strike)
        or strike <= 0
        or not np.isfinite(volatility)
        or volatility <= 0
        or not np.isfinite(time_to_expiry)
    ):
        return {
            "Delta": np.nan,
            "Gamma": np.nan,
            "Vega": np.nan,
            "Theta": np.nan,
            "Rho": np.nan,
        }

    # Handle expiry separately
    if time_to_expiry <= 0:

        if option_type == "call":
            if spot > strike:
                delta = 1.0
            elif spot < strike:
                delta = 0.0
            else:
                delta = 0.5

        else:
            if spot < strike:
                delta = -1.0
            elif spot > strike:
                delta = 0.0
            else:
                delta = -0.5

        return {
            "Delta": delta,
            "Gamma": np.nan if spot == strike else 0.0,
            "Vega": 0.0,
            "Theta": 0.0,
            "Rho": 0.0,
        }

    sqrt_time = np.sqrt(time_to_expiry)

    d1 = (
        np.log(spot / strike)
        + (
            risk_free_rate
            - dividend_yield
            + 0.5 * volatility**2
        ) * time_to_expiry
    ) / (
        volatility * sqrt_time
    )

    d2 = (
        d1
        - volatility * sqrt_time
    )

    discounted_spot = np.exp(
        -dividend_yield * time_to_expiry
    )

    discounted_strike = np.exp(
        -risk_free_rate * time_to_expiry
    )

    normal_density = norm.pdf(d1)

    # Same gamma and vega for calls and puts
    gamma = (
        discounted_spot
        * normal_density
        / (
            spot
            * volatility
            * sqrt_time
        )
    )

    raw_vega = (
        spot
        * discounted_spot
        * normal_density
        * sqrt_time
    )

    # Per one volatility percentage point
    vega = raw_vega / 100

    if option_type == "call":

        delta = (
            discounted_spot
            * norm.cdf(d1)
        )

        annual_theta = (
            -(
                spot
                * discounted_spot
                * normal_density
                * volatility
            )
            / (
                2 * sqrt_time
            )
            - risk_free_rate
            * strike
            * discounted_strike
            * norm.cdf(d2)
            + dividend_yield
            * spot
            * discounted_spot
            * norm.cdf(d1)
        )

        raw_rho = (
            strike
            * time_to_expiry
            * discounted_strike
            * norm.cdf(d2)
        )

    else:

        delta = (
            discounted_spot
            * (
                norm.cdf(d1) - 1
            )
        )

        annual_theta = (
            -(
                spot
                * discounted_spot
                * normal_density
                * volatility
            )
            / (
                2 * sqrt_time
            )
            + risk_free_rate
            * strike
            * discounted_strike
            * norm.cdf(-d2)
            - dividend_yield
            * spot
            * discounted_spot
            * norm.cdf(-d1)
        )

        raw_rho = (
            -strike
            * time_to_expiry
            * discounted_strike
            * norm.cdf(-d2)
        )

    # Per calendar day
    theta = annual_theta / 365

    # Per one interest-rate percentage point
    rho = raw_rho / 100

    return {
        "Delta": float(delta),
        "Gamma": float(gamma),
        "Vega": float(vega),
        "Theta": float(theta),
        "Rho": float(rho),
    }


def add_greek_columns(
    options,
    ticker,
    option_type,
):
    """Add analytical Black-Scholes Greeks to an option table."""

    result = options.copy()

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    # Prefer your calculated midpoint IV.
    # Use downloaded IV if midpoint IV is unavailable.
    # Use forecast RV only as a final fallback.
    result["Greek Volatility"] = (
        result["IV_Mid"]
        .fillna(result["IV Used"])
        .fillna(result["ForeV Used"])
    )

    result["Greek Volatility Source"] = np.select(
        [
            result["IV_Mid"].notna(),
            result["IV Used"].notna(),
            result["ForeV Used"].notna(),
        ],
        [
            "Calculated Mid IV",
            "Downloaded IV",
            "Forecast RV",
        ],
        default="No Volatility",
    )

    greek_results = []

    for strike, volatility in zip(
        result["strike"],
        result["Greek Volatility"],
    ):

        if (
            pd.isna(strike)
            or pd.isna(volatility)
        ):
            greeks = {
                "Delta": np.nan,
                "Gamma": np.nan,
                "Vega": np.nan,
                "Theta": np.nan,
                "Rho": np.nan,
            }

        else:
            greeks = black_scholes_greeks(
                ticker=ticker,
                strike=float(strike),
                volatility=float(volatility),
                option_type=option_type,
            )

        greek_results.append(greeks)

    greek_table = pd.DataFrame(
        greek_results,
        index=result.index,
    )

    for greek in [
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]:
        result[greek] = greek_table[greek]

    result = add_greek_sign_checks(
        options=result,
        option_type=option_type,
    )

    return result


def add_greek_sign_checks(
    options,
    option_type,
    tolerance=1e-10,
):
    """Check reliable theoretical Greek signs and ranges."""

    result = options.copy()

    option_type = option_type.lower()

    greek_columns = [
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]

    complete_greeks = (
        result[greek_columns]
        .notna()
        .all(axis=1)
    )

    result["Greek Signs Valid"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    if option_type == "call":

        valid_signs = (
            result["Delta"].between(
                -tolerance,
                1 + tolerance,
            )
            & result["Gamma"].ge(-tolerance)
            & result["Vega"].ge(-tolerance)
            & result["Rho"].ge(-tolerance)
        )

    elif option_type == "put":

        valid_signs = (
            result["Delta"].between(
                -1 - tolerance,
                tolerance,
            )
            & result["Gamma"].ge(-tolerance)
            & result["Vega"].ge(-tolerance)
            & result["Rho"].le(tolerance)
        )

    else:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    result.loc[
        complete_greeks,
        "Greek Signs Valid"
    ] = valid_signs.loc[complete_greeks]

    result["Greek Sign Issue"] = ""

    if option_type == "call":

        result.loc[
            complete_greeks
            & ~result["Delta"].between(
                -tolerance,
                1 + tolerance,
            ),
            "Greek Sign Issue",
        ] = "Call delta outside 0 to 1"

        result.loc[
            complete_greeks
            & result["Rho"].lt(-tolerance),
            "Greek Sign Issue",
        ] = "Call rho is negative"

    else:

        result.loc[
            complete_greeks
            & ~result["Delta"].between(
                -1 - tolerance,
                tolerance,
            ),
            "Greek Sign Issue",
        ] = "Put delta outside -1 to 0"

        result.loc[
            complete_greeks
            & result["Rho"].gt(tolerance),
            "Greek Sign Issue",
        ] = "Put rho is positive"

    result.loc[
        complete_greeks
        & result["Gamma"].lt(-tolerance),
        "Greek Sign Issue",
    ] = "Gamma is negative"

    result.loc[
        complete_greeks
        & result["Vega"].lt(-tolerance),
        "Greek Sign Issue",
    ] = "Vega is negative"

    result.loc[
        ~complete_greeks,
        "Greek Sign Issue",
    ] = "Missing Greek input"

    return result


def numerical_black_scholes_greeks(
    ticker,
    strike,
    volatility,
    option_type,
    spot_bump=None,
    volatility_bump=1e-4,
    rate_bump=1e-5,
    time_bump=1 / 365,
):
    """Calculate numerical Greeks using finite differences."""

    option_type = option_type.lower()

    spot = float(ticker.current_price)
    strike = float(strike)
    volatility = float(volatility)

    time_to_expiry = float(
        ticker.time_to_expiry
    )

    risk_free_rate = float(
        ticker.risk_free_rate
    )

    dividend_yield = float(
        ticker.dividend_yield
    )

    if (
        spot <= 0
        or strike <= 0
        or volatility <= 0
        or time_to_expiry <= 0
    ):
        return {
            "Numerical Delta": np.nan,
            "Numerical Gamma": np.nan,
            "Numerical Vega": np.nan,
            "Numerical Theta": np.nan,
            "Numerical Rho": np.nan,
        }

    if spot_bump is None:
        spot_bump = max(
            spot * 1e-4,
            1e-4,
        )

    # Prevent negative spot
    spot_bump = min(
        spot_bump,
        spot * 0.5,
    )

    # Prevent negative volatility
    volatility_bump = min(
        volatility_bump,
        volatility * 0.5,
    )

    # Keep the shorter time positive
    time_bump = min(
        time_bump,
        time_to_expiry * 0.5,
    )

    def price(
        adjusted_spot=spot,
        adjusted_volatility=volatility,
        adjusted_rate=risk_free_rate,
        adjusted_time=time_to_expiry,
    ):
        return scalar_black_scholes_price(
            ticker=ticker,
            spot=adjusted_spot,
            strike=strike,
            time_to_expiry=adjusted_time,
            risk_free_rate=adjusted_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            volatility=adjusted_volatility,
        )

    base_price = price()

    price_spot_up = price(
        adjusted_spot=spot + spot_bump,
    )

    price_spot_down = price(
        adjusted_spot=spot - spot_bump,
    )

    numerical_delta = (
        price_spot_up
        - price_spot_down
    ) / (
        2 * spot_bump
    )

    numerical_gamma = (
        price_spot_up
        - 2 * base_price
        + price_spot_down
    ) / (
        spot_bump**2
    )

    price_vol_up = price(
        adjusted_volatility=(
            volatility + volatility_bump
        ),
    )

    price_vol_down = price(
        adjusted_volatility=(
            volatility - volatility_bump
        ),
    )

    # First calculate derivative per 1.00 volatility,
    # then convert to one percentage point
    numerical_vega = (
        (
            price_vol_up
            - price_vol_down
        )
        / (
            2 * volatility_bump
        )
        * 0.01
    )

    price_rate_up = price(
        adjusted_rate=(
            risk_free_rate + rate_bump
        ),
    )

    price_rate_down = price(
        adjusted_rate=(
            risk_free_rate - rate_bump
        ),
    )

    # Convert to one percentage-point rate change
    numerical_rho = (
        (
            price_rate_up
            - price_rate_down
        )
        / (
            2 * rate_bump
        )
        * 0.01
    )

    price_less_time = price(
        adjusted_time=(
            time_to_expiry - time_bump
        ),
    )

    price_more_time = price(
        adjusted_time=(
            time_to_expiry + time_bump
        ),
    )

    # Theta is the effect of calendar time passing.
    # This is the negative derivative with respect to T.
    numerical_theta = (
        price_less_time
        - price_more_time
    ) / (
        2
        * time_bump
        * 365
    )

    return {
        "Numerical Delta": float(numerical_delta),
        "Numerical Gamma": float(numerical_gamma),
        "Numerical Vega": float(numerical_vega),
        "Numerical Theta": float(numerical_theta),
        "Numerical Rho": float(numerical_rho),
    }


def add_numerical_greek_validation(
    options,
    ticker,
    option_type,
):
    """Compare analytical Greeks with finite-difference Greeks."""

    result = options.copy()

    numerical_results = []

    for strike, volatility in zip(
        result["strike"],
        result["Greek Volatility"],
    ):

        if (
            pd.isna(strike)
            or pd.isna(volatility)
        ):
            numerical_greeks = {
                "Numerical Delta": np.nan,
                "Numerical Gamma": np.nan,
                "Numerical Vega": np.nan,
                "Numerical Theta": np.nan,
                "Numerical Rho": np.nan,
            }

        else:
            numerical_greeks = (
                numerical_black_scholes_greeks(
                    ticker=ticker,
                    strike=float(strike),
                    volatility=float(volatility),
                    option_type=option_type,
                )
            )

        numerical_results.append(
            numerical_greeks
        )

    numerical_table = pd.DataFrame(
        numerical_results,
        index=result.index,
    )

    for greek in [
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]:
        numerical_column = (
            f"Numerical {greek}"
        )

        result[numerical_column] = (
            numerical_table[
                numerical_column
            ]
        )

        result[f"{greek} Error"] = (
            result[greek]
            - result[numerical_column]
        )

        result[f"{greek} Absolute Error"] = (
            result[f"{greek} Error"]
            .abs()
        )

        settings = (
            GREEK_VALIDATION_TOLERANCES[
                greek
            ]
        )

        valid_comparison = (
            result[greek].notna()
            & result[numerical_column].notna()
        )

        passed = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

        passed.loc[valid_comparison] = np.isclose(
            result.loc[
                valid_comparison,
                greek,
            ],
            result.loc[
                valid_comparison,
                numerical_column,
            ],
            atol=settings["atol"],
            rtol=settings["rtol"],
        )

        result[
            f"{greek} Validation Passed"
        ] = passed

    validation_columns = [
        f"{greek} Validation Passed"
        for greek in [
            "Delta",
            "Gamma",
            "Vega",
            "Theta",
            "Rho",
        ]
    ]

    complete_validation = (
        result[validation_columns]
        .notna()
        .all(axis=1)
    )

    result["All Greeks Valid"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    result.loc[
        complete_validation,
        "All Greeks Valid",
    ] = (
        result.loc[
            complete_validation,
            validation_columns,
        ]
        .all(axis=1)
    )

    return result

"""GBP valuation of a USD-funded sleeve, including explicit conversion events."""

import numpy as np
import pandas as pd


def gbp_account_returns(
    frame,
    *,
    fx_file,
    initial_date,
    initial_capital_gbp=4000,
    fx_conversion_fee=0.0015,
    convert_back_at_end=True,
):
    """Value all stock and idle USD capital in GBP; no intermediate conversions.

    FX file contains Date and GBP_per_USD. Exact observation-date matches are
    required. Daily closing FX is a valuation proxy, not an execution quote.
    """
    if not np.isfinite(initial_capital_gbp) or initial_capital_gbp <= 0:
        raise ValueError("initial_capital_gbp must be positive")
    if not np.isfinite(fx_conversion_fee) or not 0 <= fx_conversion_fee < 1:
        raise ValueError("fx_conversion_fee must lie in [0, 1)")
    if not isinstance(convert_back_at_end, bool):
        raise ValueError("convert_back_at_end must be boolean")
    data = frame.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(data["Date"], errors="raise"))
    initial = pd.Timestamp(initial_date)
    if (
        len(dates) == 0
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or initial >= dates[0]
    ):
        raise ValueError("Account requires ordered unique returns after initial_date")
    fx = pd.read_csv(fx_file)
    fx["Date"] = pd.to_datetime(fx["Date"], errors="raise")
    if fx["Date"].isna().any() or fx["Date"].duplicated().any():
        raise ValueError("FX dates must be unique and nonmissing")
    rates = pd.to_numeric(fx.set_index("Date")["GBP_per_USD"], errors="raise")
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("FX rates must be finite and positive")
    aligned = rates.reindex(pd.DatetimeIndex([initial]).append(dates))
    if aligned.isna().any():
        raise ValueError(
            "FX file missing required dates: "
            + ", ".join(str(x.date()) for x in aligned.index[aligned.isna()][:5])
        )
    usd = pd.to_numeric(data["Return"], errors="raise").to_numpy()
    if not np.isfinite(usd).all() or (usd < -1).any():
        raise ValueError("USD returns must be finite and at least -1")
    factors = (1 + usd) * aligned.to_numpy()[1:] / aligned.to_numpy()[:-1]
    factors[0] *= 1 - fx_conversion_fee
    if convert_back_at_end:
        factors[-1] *= 1 - fx_conversion_fee
    data["USD Return"] = usd
    data["Return"] = factors - 1
    data["GBP Equity"] = initial_capital_gbp * np.cumprod(factors)
    return data

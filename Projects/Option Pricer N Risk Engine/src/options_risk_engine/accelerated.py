
"""Optional access to the pybind11 extension with a safe Python fallback."""

from __future__ import annotations

import numpy as np

from options_risk_engine.pricing.black_scholes import black_scholes

try:
    import fast_options as _fast_options
    HAS_CPP = True
except ImportError:
    _fast_options = None
    HAS_CPP = False


def black_scholes_batch(spot, strike, time, rate, dividend_yield, volatility, is_call):
    arrays = [np.asarray(value, dtype=float) for value in (
        spot, strike, time, rate, dividend_yield, volatility
    )]
    arrays = np.broadcast_arrays(*arrays)
    arrays = [array.ravel() for array in arrays]
    if HAS_CPP:
        return _fast_options.black_scholes_batch(*arrays, bool(is_call))
    return np.asarray(black_scholes(
        ticker=None,
        spot=arrays[0], strike=arrays[1], time_to_expiry=arrays[2],
        risk_free_rate=arrays[3], dividend_yield=arrays[4],
        volatility=arrays[5], option_type="call" if is_call else "put",
    ))


import numpy as np
import pandas as pd
from options_risk_engine.domain import OptionTicker
from options_risk_engine.pricing.greeks import black_scholes_greeks, numerical_black_scholes_greeks


def test_analytical_greeks_match_finite_difference():
    ticker = OptionTicker("TEST", valuation_date=pd.Timestamp("2026-01-01"))
    ticker.current_price = 100.0
    ticker.time_to_expiry = 0.5
    ticker.risk_free_rate = 0.04
    ticker.dividend_yield = 0.01
    analytical = black_scholes_greeks(ticker, 100.0, 0.25, "call")
    numerical = numerical_black_scholes_greeks(ticker, 100.0, 0.25, "call")
    for greek in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
        assert np.isclose(analytical[greek], numerical[f"Numerical {greek}"], rtol=2e-2, atol=2e-3)

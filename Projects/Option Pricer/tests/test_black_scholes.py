
import numpy as np
from options_risk_engine.pricing.black_scholes import black_scholes


def test_call_put_parity():
    s, k, t, r, q, sigma = 100.0, 105.0, 0.5, 0.04, 0.01, 0.25
    call = black_scholes(None, s, k, t, r, q, "call", sigma)
    put = black_scholes(None, s, k, t, r, q, "put", sigma)
    expected = s * np.exp(-q * t) - k * np.exp(-r * t)
    assert np.isclose(call - put, expected, atol=1e-10)


def test_expiry_uses_intrinsic_value():
    assert black_scholes(None, 110, 100, 0, 0.03, 0, "call", 0.2) == 10
    assert black_scholes(None, 90, 100, 0, 0.03, 0, "put", 0.2) == 10

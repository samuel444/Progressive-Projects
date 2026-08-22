
import numpy as np
import pandas as pd
from options_risk_engine.risk.attribution import calculate_greek_profit_loss


def test_base_attribution_is_zero():
    expanded = pd.DataFrame({
        "Scenario ID": ["BASE"],
        "Contract Symbol": ["SHARE_TEST"],
        "Ticker_x": ["TEST"],
        "Option Type": ["Shares"],
        "Current Stock Price": [100.0],
        "Shocked Spot": [100.0],
        "Scenario PnL": [0.0],
        "Position Delta": [10.0],
        "Position Gamma": [0.0],
        "Position Vega": [0.0],
        "Position Theta": [0.0],
        "Position Rho": [0.0],
        "Spot Shock": [0.0],
        "Volatility Shock": [0.0],
        "Rate Shock": [0.0],
        "Days Forward": [0],
    })
    _, scenario = calculate_greek_profit_loss(expanded)
    zero_columns = [
        "Full_Revaluation_PnL", "Delta_PnL", "Gamma_PnL", "Vega_PnL",
        "Theta_PnL", "Rho_PnL", "Approximate_PnL", "Residual_PnL",
        "Gross_Full_PnL", "Gross_Residual", "Attribution Check",
    ]
    assert np.allclose(scenario[zero_columns].to_numpy(), 0.0)

import yfinance as yf
import numpy as np
from scipy.optimize import minimize

tickers = ["AAPL", "MSFT", "NVDA", "GOOG"]

# Download prices
prices = yf.download(
    tickers,
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)["Close"]

# Daily returns
returns = prices.pct_change().dropna()

# Annualised expected returns and covariance
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252


def negative_sharpe(weights):
    portfolio_return = weights @ mean_returns

    portfolio_volatility = np.sqrt(
        weights.T @ cov_matrix @ weights
    )

    sharpe = (
        portfolio_return
    ) / portfolio_volatility

    # scipy minimises, so minimise negative Sharpe
    return -sharpe

n_assets = len(tickers)

# Start equally weighted
initial_weights = np.ones(n_assets) / n_assets

# Weights must add to 1
constraints = {
    "type": "eq",
    "fun": lambda weights: np.sum(weights) - 1
}

# Long-only: each weight between 0 and 1
bounds = [(0, 1)] * n_assets

result = minimize(
    negative_sharpe,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x

portfolio_return = optimal_weights @ mean_returns
portfolio_volatility = np.sqrt(
    optimal_weights.T @ cov_matrix @ optimal_weights
)
portfolio_sharpe = (
    portfolio_return
) / portfolio_volatility

print("Maximum Sharpe Portfolio:")

for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

print(f"\nExpected Return: {portfolio_return:.2%}")
print(f"Volatility: {portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {portfolio_sharpe:.3f}")
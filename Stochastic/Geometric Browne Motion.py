import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Download Apple stock data
# --------------------------------

df = yf.download(
    "AAPL",
    period="1y",
    auto_adjust=True,
    progress=False
)

df = df[["Close"]]

df["Log Returns"] = np.log(df["Close"] / (df["Close"].shift(1)))

df = df.dropna()

prices = df["Close"]

log_returns = df["Log Returns"].values

# Daily estimates
daily_mean = np.mean(log_returns)
daily_std = np.std(log_returns, ddof=1)

# Annualised GBM parameters
sigma = daily_std * np.sqrt(252)

mu = daily_mean * 252 + 0.5 * sigma**2

print(f"Drift (mu): {mu:.3f}")
print(f"Diffusion / volatility (sigma): {sigma:.3f}")


# Brownian motion
n = len(df)

dt = 1 / 252

dW = np.sqrt(dt) * np.random.randn(n - 1)

W = np.concatenate([
    [0],
    np.cumsum(dW)
])

t = np.arange(n) * dt


# Geometric Brownian Motion
S0 = prices.values[0]

simulated_prices = S0 * np.exp(
    (mu - 0.5 * sigma**2) * t
    + sigma * W
)


# Plot actual vs simulated Apple
plt.plot(
    prices.index,
    prices,
    label="Actual AAPL"
)

plt.plot(
    prices.index,
    simulated_prices,
    label="GBM Simulation"
)

plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.title("Apple Stock - Geometric Brownian Motion")

plt.legend()
plt.show()
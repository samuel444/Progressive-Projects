import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# Download Apple stock data
df = yf.download(
    "AAPL",
    period="1y",
    auto_adjust=True,
    progress=False
)

# Make a simple dataframe with Close
df = df[["Close"]]

# Log returns
df["Log Returns"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

df = df.dropna()

prices = df["Close"]
log_returns = df["Log Returns"].values


# Estimate drift and volatility
daily_mean = np.mean(log_returns)
daily_std = np.std(log_returns, ddof=1)

# Annualised GBM parameters
sigma = daily_std * np.sqrt(252)

mu = daily_mean * 252 + 0.5 * sigma**2

dt = 1 / 252

# Start from today's price
S0 = prices.values[-1]


# Monte Carlo settings
days = 252
simulations = 1000

# Time for one future year
t = np.arange(days + 1) * dt

# Store every simulation
paths = np.zeros((days + 1, simulations))


# Monte Carlo simulation
for i in range(simulations):

    # Brownian motion
    dW = np.sqrt(dt) * np.random.randn(days)

    W = np.concatenate([
        [0],
        np.cumsum(dW)
    ])

    # Geometric Brownian Motion
    paths[:, i] = S0 * np.exp(
        (mu - 0.5 * sigma**2) * t
        + sigma * W
    )


# Plot simulated paths
plt.plot(paths[:, :100])

plt.xlabel("Trading Days")
plt.ylabel("Price ($)")
plt.title("Apple Monte Carlo Simulation")

plt.show()


# Final price distribution
final_prices = paths[-1]

print(f"Mean final price: ${np.mean(final_prices):.2f}")
print(f"Median final price: ${np.median(final_prices):.2f}")
print(f"5th percentile: ${np.percentile(final_prices, 5):.2f}")
print(f"95th percentile: ${np.percentile(final_prices, 95):.2f}")

plt.hist(final_prices, bins=50)

plt.xlabel("Final Price ($)")
plt.ylabel("Frequency")
plt.title("Distribution of Final Apple Prices")

plt.show()
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# Download Apple data
df = yf.download(
    "AAPL",
    period="1y",
    auto_adjust=True,
    progress=False
)

df = df["Close"].squeeze().to_frame(name="Close")

df["Log Returns"] = np.log(
    df["Close"] / df["Close"].shift(1)
)

df = df.dropna()

log_returns = df["Log Returns"].values


# Estimate GBM parameters
daily_mean = np.mean(log_returns)
daily_std = np.std(log_returns, ddof=1)

sigma = daily_std * np.sqrt(252)

mu = daily_mean * 252 + 0.5 * sigma**2

print(f"Drift: {mu:.3f}")
print(f"Volatility: {sigma:.3f}")


# Future simulation settings
S0 = df["Close"].iloc[-1]

days = 60
simulations = 1000

dt = 1 / 252

paths = np.zeros((days + 1, simulations))

paths[0] = S0


# --------------------------------
# Euler-Maruyama
# --------------------------------

for t in range(days):

    Z = np.random.randn(simulations)

    paths[t + 1] = (
        paths[t]
        + mu * paths[t] * dt
        + sigma * paths[t] * np.sqrt(dt) * Z
    )


# Forecast distribution
median_path = np.median(paths, axis=1)

lower = np.percentile(paths, 5, axis=1)
upper = np.percentile(paths, 95, axis=1)


# Plot simulations
plt.plot(paths[:, :50], alpha=0.4)

plt.plot(
    median_path,
    linewidth=3,
    label="Median simulation"
)

plt.xlabel("Trading Days Ahead")
plt.ylabel("Price ($)")
plt.title("AAPL Euler-Maruyama Simulations")

plt.legend()
plt.show()


# Final-day results
print(f"Current price: ${S0:.2f}")
print(f"Median price in {days} days: ${median_path[-1]:.2f}")
print(f"5th percentile: ${lower[-1]:.2f}")
print(f"95th percentile: ${upper[-1]:.2f}")
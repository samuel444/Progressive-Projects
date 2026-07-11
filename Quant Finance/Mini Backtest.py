import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True,
    progress=False
)


# Keep only the closing price
df = df[["Close"]]


# Calculate daily returns
df["Return"] = df["Close"].pct_change()


# Calculate the short-term moving average
df["MA_20"] = (
    df["Close"]
    .rolling(window=20)
    .mean()
)


# Calculate the long-term moving average
df["MA_50"] = (
    df["Close"]
    .rolling(window=50)
    .mean()
)


# Create the trading signal
# 1 means invested in Apple
# 0 means out of the market
df["Signal"] = np.where(
    df["MA_20"] < df["MA_50"],
    1,
    0
)


# Shift the signal to avoid lookahead bias
# Today's position must be based on information available yesterday
df["Position"] = df["Signal"].shift(1)


# Calculate the strategy returns
df["Strategy_Return"] = (
    df["Position"]
    * df["Return"]
)


# Remove missing values
df = df.dropna()


# Calculate cumulative buy-and-hold returns
df["Buy_And_Hold"] = (
    1 + df["Return"]
).cumprod()


# Calculate cumulative strategy returns
df["Strategy"] = (
    1 + df["Strategy_Return"]
).cumprod()


# Display the backtest data
print(df.head())


# Plot the price and moving averages
df[["Close", "MA_20", "MA_50"]].plot(
    title="Apple Moving Average Strategy"
)

plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# Compare the strategy against buy and hold
df[["Buy_And_Hold", "Strategy"]].plot(
    title="Strategy vs Buy and Hold"
)

plt.xlabel("Date")
plt.ylabel("Growth of £1")
plt.show()


# Calculate total returns
buy_and_hold_return = (
    df["Buy_And_Hold"].iloc[-1] - 1
)

strategy_return = (
    df["Strategy"].iloc[-1] - 1
)


# Calculate annualised volatility
buy_and_hold_volatility = (
    df["Return"].std()
    * np.sqrt(252)
)

strategy_volatility = (
    df["Strategy_Return"].std()
    * np.sqrt(252)
)


# Print the backtest results
print("\nBuy and Hold Total Return:")
print(buy_and_hold_return)

print("\nStrategy Total Return:")
print(strategy_return)

print("\nBuy and Hold Annualised Volatility:")
print(buy_and_hold_volatility)

print("\nStrategy Annualised Volatility:")
print(strategy_volatility)
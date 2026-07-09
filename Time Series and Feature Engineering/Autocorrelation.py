import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = "AAPL"

df = yf.download(
    ticker,
    start="2022-01-01",
    end="2025-01-01",
    auto_adjust=True
)


# Calculate daily returns
df["Return"] = df["Close"].pct_change()


# Absolute returns (volatility)
df["Absolute Return"] = df["Return"].abs()


# Autocorrelation calculations
print("Lag 1 Return Autocorrelation")
print(df["Return"].autocorr(lag=1))

print("\nLag 5 Return Autocorrelation")
print(df["Return"].autocorr(lag=5))

print("\nLag 1 Absolute Return Autocorrelation")
print(df["Absolute Return"].autocorr(lag=1))

print("\nLag 5 Absolute Return Autocorrelation")
print(df["Absolute Return"].autocorr(lag=5))


# Plot returns
plt.plot(df.index, df["Return"])
plt.title("Apple Daily Returns")
plt.ylabel("Return")
plt.grid(True)
plt.show()


# Plot absolute returns
plt.plot(df.index, df["Absolute Return"])
plt.title("Absolute Daily Returns")
plt.ylabel("Absolute Return")
plt.grid(True)
plt.show()
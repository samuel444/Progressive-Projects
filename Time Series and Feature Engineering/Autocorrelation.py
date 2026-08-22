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


# Autocorrelation calculations
print("Lag 1 Return Autocorrelation")
print(df["Return"].autocorr(lag=1))

print("\nLag 5 Return Autocorrelation")
print(df["Return"].autocorr(lag=5))


# Absolute returns (volatility)
df["Absolute Return"] = df["Return"].abs()

print("\nLag 1 Absolute Return Autocorrelation")
print(df["Absolute Return"].autocorr(lag=1))

print("\nLag 5 Absolute Return Autocorrelation")
print(df["Absolute Return"].autocorr(lag=5))



df["Relative Volume"] = (
    df["Volume"] /
    df["Volume"].rolling(30).mean()
)

print("\nLag 1 Relative Volume Autocorrelation")
print(df["Relative Volume"].autocorr(lag=1))

print("\nLag 5 Relative Volume Autocorrelation")
print(df["Relative Volume"].autocorr(lag=5))





import yfinance as yf
import numpy as np

# Download Apple stock data
df = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True,
    progress=False
)

# Keep only the required columns
df = df[["Open", "High", "Low", "Close", "Volume"]]


# Calculate daily returns
df["Return"] = df["Close"].pct_change()


# Create lagged returns
df["Lag_1_Return"] = df["Return"].shift(1)
df["Lag_5_Return"] = df["Return"].shift(5)


# Calculate 20-day annualised realised volatility
df["RV_20"] = (
    df["Return"]
    .rolling(window=20)
    .std()
    * np.sqrt(252)
)


# Calculate 20-day momentum
df["Momentum_20"] = (
    df["Close"]
    / df["Close"].shift(20)
    - 1
)


# Calculate the relative trading volume
df["Relative_Volume"] = (
    df["Volume"]
    / df["Volume"]
    .rolling(window=20)
    .mean()
)

# Calculate rolling skewness
df["Rolling_Skew"] = (
    df["Return"]
    .rolling(window=20)
    .skew()
)


# Calculate rolling kurtosis
df["Rolling_Kurtosis"] = (
    df["Return"]
    .rolling(window=20)
    .kurt()
)


# Remove missing values created by rolling windows
df = df.dropna()


# Display the engineered features
print(df.head())

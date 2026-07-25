import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True,
    progress=False
)
df = df[["Close"]]

# Plot the price at close
df["Close"].plot(title="Apple Closing Price")

plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Calculate returns
df["Return"] = df["Close"].pct_change()

print(df.head())

# Plot the returns
df["Return"].plot(title="Apple Daily Returns")

plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.show()


# Rolling mean of price
df["Price_Rolling_Mean"] = (
    df["Close"]
    .rolling(window=30)
    .mean()
)

df["Price_Rolling_Mean"].plot(title="Rolling Mean of Price")

plt.xlabel("Date")
plt.ylabel("Rolling Average Price")
plt.show()


# Rolling mean of returns
df["Return_Rolling_Mean"] = (
    df["Return"]
    .rolling(window=30)
    .mean()
)

df["Return_Rolling_Mean"].plot(title="Rolling Mean of Returns")

plt.xlabel("Date")
plt.ylabel("Rolling Average Return")
plt.show()


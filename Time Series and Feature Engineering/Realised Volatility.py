import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download(
    "AAPL",
    start="2020-01-01",
    auto_adjust=True
)

df["Return"] = df["Close"].pct_change()

# Annualised realised volatility
df["20 Day Realised Volatility"] = (
    df["Return"]
    .rolling(20)
    .std()
    * (252 ** 0.5)
)

df["60 Day RV"] = df["Return"].rolling(60).std() * (252 ** 0.5)
df["252 Day RV"] = df["Return"].rolling(252).std() * (252 ** 0.5)

print(df[["Return", "20 Day Realised Volatility"]])

plt.plot(df.index, df["20 Day Realised Volatility"])
plt.title("20-Day Annualised Realised Volatility")
plt.ylabel("Volatility")
plt.grid(True)
plt.show()
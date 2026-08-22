import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download(
    "AAPL",
    start="2020-01-01",
    auto_adjust=True
)

df["Return"] = df["Close"].pct_change()

# Realised Volatility
df["20 Day Realised Volatility"] = (
    df["Return"]
    .rolling(20)
    .std()
)

# Annualised realised volatility
df["Annualised 20 Day Realised Volatility"] = (
    df["20 Day Realised Volatility"] * (252 ** 0.5)
)

print(df[["Return", "Annualised 20 Day Realised Volatility"]])

plt.plot(df.index, df["Annualised 20 Day Realised Volatility"])
plt.title("20-Day Annualised Realised Volatility")
plt.ylabel("Volatility")
plt.grid(True)
plt.show()




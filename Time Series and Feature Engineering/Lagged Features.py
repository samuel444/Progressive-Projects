import pandas as pd

import yfinance as yf

apple = yf.download("AAPL", period="400d", interval="1d")

apple["Yesterday"] = apple["Close"].shift(1)
apple["Last Month"] = apple["Close"].shift(30)
apple["Last Year"] = apple["Close"].shift(365)

apple["Return"] = apple["Close"].pct_change()
apple["Return Yesterday"] = apple["Return"].shift(1)

apple = apple.dropna()

print("\nApple Last 30 Days Data:")
print(apple[["Close", "Yesterday", "Last Month", "Last Year", "Return", "Return Yesterday"]].tail(30))
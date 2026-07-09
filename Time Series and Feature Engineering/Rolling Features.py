import pandas as pd

import yfinance as yf

apple = yf.download("AAPL", period="400d", interval="1d")

apple["Rolling Week Mean"] = apple["Close"].rolling(window=7).mean()
apple["Rolling Quarter Median"] = apple["Close"].rolling(window=90).median()
apple["Rolling Month Std"] = apple["Close"].rolling(window=30).std()
apple["Rolling Year Min"] = apple["Close"].rolling(window=365).min()
apple["Rolling Max"] = apple["Close"].expanding().max()

apple = apple.dropna()


print("\nApple Data:")
print(apple[["Close", "Rolling Week Mean", "Rolling Quarter Median", "Rolling Month Std", "Rolling Year Min", "Rolling Max"]])
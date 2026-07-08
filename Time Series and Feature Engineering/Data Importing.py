import pandas as pd

# df = pd.read_csv("apple_prices.csv")

import yfinance as yf

apple = yf.download("AAPL",start="2025-01-01",
    end="2026-01-01")

print("\nApple 2025-2026 Data:")
print(apple)


microsoft = yf.download("MSFT", period="3d", interval="1h")

print("\nMicrosoft Hourly Data:")
print(microsoft)


google = yf.download(["GOOGL", "MSFT", "AAPL"], start="2025-01-01", end="2026-01-01")
google = google["Close","MSFT"]
print("\nGoogle Close Prices:")
print(google)


print('\nMicrosoft Percent Change:')
print(microsoft["Close"].pct_change())


apple.to_csv("/Users/sam/Progressive-Projects/Time Series and Feature Engineering/apple_prices.csv")


apple = pd.read_csv(
    "/Users/sam/Progressive-Projects/Time Series and Feature Engineering/apple_prices.csv"
)

print('\nApple DataFrame from CSV:')
print(apple)


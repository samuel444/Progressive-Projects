import yfinance as yf
import numpy as np

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

# /////////////////////////////////

years = (
    df.index[-1] - df.index[0]
).days / 365.25


strategy_annual_return = (
    df["Strategy"].iloc[-1]
    ** (1 / years)
    - 1
)

buy_hold_annual_return = (
    df["Buy_And_Hold"].iloc[-1]
    ** (1 / years)
    - 1
)


# Sharpe Ratio
strategy_sharpe = (
    strategy_annual_return
    / strategy_volatility
)

buy_hold_sharpe = (
    buy_hold_annual_return
    / buy_and_hold_volatility
)


# Running maximum
df["Buy_And_Hold_Peak"] = (
    df["Buy_And_Hold"]
    .cummax()
)

df["Strategy_Peak"] = (
    df["Strategy"]
    .cummax()
)


# Drawdown
df["Buy_And_Hold_Drawdown"] = (
    (df["Buy_And_Hold"] - df["Buy_And_Hold_Peak"])
    / df["Buy_And_Hold_Peak"]
)

df["Strategy_Drawdown"] = (
    (df["Strategy"] - df["Strategy_Peak"])
    / df["Strategy_Peak"]
)


# Maximum drawdown
buy_hold_max_drawdown = (
    df["Buy_And_Hold_Drawdown"].min()
)

strategy_max_drawdown = (
    df["Strategy_Drawdown"].min()
)


# Display the results
# Print the backtest results
print("\nBuy and Hold:")
print("Return:",buy_and_hold_return)
print("Sharpe Ratio:", buy_hold_sharpe)
print("Maximum Drawdown:", buy_hold_max_drawdown)

print("\nStrategy:")
print("Return:", strategy_return)
print("Sharpe Ratio:", strategy_sharpe)
print("Maximum Drawdown:", strategy_max_drawdown)

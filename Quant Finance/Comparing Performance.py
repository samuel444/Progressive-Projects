import yfinance as yf
import pandas as pd
import numpy as np

# Download Apple stock data
df = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2026-07-12",
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
# 1 means the strategy wants to own Apple
# 0 means the strategy wants to stay out of the market
df["Signal"] = np.where(
    df["MA_20"] > df["MA_50"],
    1,
    0
)


# Shift the signal to avoid lookahead bias
# Today's position is based on yesterday's signal
df["Position"] = df["Signal"].shift(1)


# Calculate the strategy returns before trading fees
df["Strategy_Return_Before_Fees"] = (
    df["Position"]
    * df["Return"]
)


# Identify when the position changes
# A value of 1 means the strategy entered or exited a position
df["Trade"] = (
    df["Position"]
    .diff()
    .abs()
)


# Set the estimated cost per trade
# 0.001 means a trading cost of 0.1%
trading_cost = 0.001


# Calculate the trading cost paid each day
df["Trading_Cost"] = (
    df["Trade"]
    * trading_cost
)


# Calculate strategy returns after trading fees
df["Strategy_Return"] = (
    df["Strategy_Return_Before_Fees"]
    - df["Trading_Cost"]
)


# Remove missing values
df = df.dropna()


# Calculate cumulative buy-and-hold growth
df["Buy_And_Hold"] = (
    1 + df["Return"]
).cumprod()


# Calculate cumulative strategy growth before fees
df["Strategy_Before_Fees"] = (
    1 + df["Strategy_Return_Before_Fees"]
).cumprod()


# Calculate cumulative strategy growth after fees
df["Strategy"] = (
    1 + df["Strategy_Return"]
).cumprod()



# Calculate total returns
buy_and_hold_return = (
    df["Buy_And_Hold"].iloc[-1]
    - 1
)

strategy_return_before_fees = (
    df["Strategy_Before_Fees"].iloc[-1]
    - 1
)

strategy_return = (
    df["Strategy"].iloc[-1]
    - 1
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


# Calculate annualised mean returns
buy_and_hold_annual_return = (
    df["Return"].mean()
    * 252
)

strategy_annual_return = (
    df["Strategy_Return"].mean()
    * 252
)




# Calculate Sharpe ratios
buy_and_hold_sharpe = (
    buy_and_hold_annual_return
) / buy_and_hold_volatility

strategy_sharpe = (
    strategy_annual_return
) / strategy_volatility


# Calculate the running maximum values
df["Buy_And_Hold_Peak"] = (
    df["Buy_And_Hold"]
    .cummax()
)

df["Strategy_Peak"] = (
    df["Strategy"]
    .cummax()
)


# Calculate drawdowns
df["Buy_And_Hold_Drawdown"] = (
    (
        df["Buy_And_Hold"]
        - df["Buy_And_Hold_Peak"]
    )
    / df["Buy_And_Hold_Peak"]
)

df["Strategy_Drawdown"] = (
    (
        df["Strategy"]
        - df["Strategy_Peak"]
    )
    / df["Strategy_Peak"]
)


# Calculate maximum drawdowns
buy_and_hold_max_drawdown = (
    df["Buy_And_Hold_Drawdown"]
    .min()
)

strategy_max_drawdown = (
    df["Strategy_Drawdown"]
    .min()
)


# Find the dates of the largest drawdowns
buy_and_hold_worst_date = (
    df["Buy_And_Hold_Drawdown"]
    .idxmin()
)

strategy_worst_date = (
    df["Strategy_Drawdown"]
    .idxmin()
)



# Create a strategy comparison table
comparison = pd.DataFrame({
    "Buy and Hold": [
        buy_and_hold_return,
        buy_and_hold_annual_return,
        buy_and_hold_volatility,
        buy_and_hold_sharpe,
        buy_and_hold_max_drawdown
    ],
    "Strategy": [
        strategy_return,
        strategy_annual_return,
        strategy_volatility,
        strategy_sharpe,
        strategy_max_drawdown
    ]
},
index=[
    "Total Return",
    "Annualised Return",
    "Annualised Volatility",
    "Sharpe Ratio",
    "Maximum Drawdown"
])


# Display the comparison table
print("\nStrategy Comparison:")
print(comparison)


# Display the effect of trading fees
print("\nStrategy Return Before Fees:")
print(strategy_return_before_fees)

print("\nStrategy Return After Fees:")
print(strategy_return)

print("\nNumber of Position Changes:")
print(df["Trade"].sum())


# Display the worst drawdown dates
print("\nBuy and Hold Worst Drawdown Date:")
print(buy_and_hold_worst_date)

print("\nStrategy Worst Drawdown Date:")
print(strategy_worst_date)


# --------------------------------------------------
# Backtest mistake 1: Lookahead bias
# --------------------------------------------------



# --------------------------------------------------
# Backtest mistake 2: Ignoring trading fees
# --------------------------------------------------


# --------------------------------------------------
# Backtest mistake 3: Overfitting
# --------------------------------------------------


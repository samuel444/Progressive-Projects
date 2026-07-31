import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import norm

def black_scholes(calls,puts,current_price, r=0.0375, sigma_calls = None):
    expiry_calls = calls[1]
    expiry_puts = puts[1]

    if sigma_calls is None:
        sigma_calls = calls[0]["impliedVolatility"] 
        sigma_puts = puts[0]["impliedVolatility"]
    else:
        sigma_puts = sigma_calls

    T_calls = (
        pd.Timestamp(expiry_calls) - pd.Timestamp.today().normalize()
    ).days / 365
    T_puts = (
            pd.Timestamp(expiry_puts) - pd.Timestamp.today().normalize()
        ).days / 365
    

    strike_calls = calls[0]["strike"]
    strike_puts = puts[0]["strike"]

    d1_calls = (
        np.log(current_price / strike_calls)
        + (r + sigma_calls**2 / 2) * T_calls
    ) / (
        sigma_calls * np.sqrt(T_calls)
    )

    d2_calls = (
        d1_calls
        - sigma_calls * np.sqrt(T_calls)
    )

    call_prices = (
        current_price * norm.cdf(d1_calls)
        - strike_calls
        * np.exp(-r * T_calls)
        * norm.cdf(d2_calls)
    )


    # -------------------------
    # PUTS
    # -------------------------

    d1_puts = (
        np.log(current_price / strike_puts)
        + (r + sigma_puts**2 / 2) * T_puts
    ) / (
        sigma_puts * np.sqrt(T_puts)
    )

    d2_puts = (
        d1_puts
        - sigma_puts * np.sqrt(T_puts)
    )

    put_prices = (
        strike_puts
        * np.exp(-r * T_puts)
        * norm.cdf(-d2_puts)
        - current_price
        * norm.cdf(-d1_puts)
    )

    return call_prices, put_prices

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

symbols = ["AAPL", "MSFT", "META", "WMT", "GOOGL", "AMZN", "HCA"]

df = yf.download(
    symbols,
    period="max",
    interval="1d",
    auto_adjust=True,
    progress=False
)

# Keep only closing prices
df = df[["Close"]]


# --------------------------------------------------
# RETURNS
# --------------------------------------------------

returns = df["Close"].pct_change()

returns.columns = pd.MultiIndex.from_product(
    [["Return"], returns.columns],
    names=df.columns.names
)

df = pd.concat([df, returns], axis=1)


# Get returns with just ticker columns
r = df["Return"]


# -----------------------------
# REALISED VOLATILITY
# Used directly in Black-Scholes
# -----------------------------

rv_20 = r.rolling(20).std() * np.sqrt(252)
rv_60 = r.rolling(60).std() * np.sqrt(252)
rv_252 = r.rolling(252).std() * np.sqrt(252)


# -----------------------------
# VOLATILITY FORECAST FEATURES
# -----------------------------

# Absolute and squared returns
# Large values indicate volatility shocks
abs_return = r.abs()
squared_return = r ** 2


# Exponentially weighted volatility
# Gives more importance to recent observations
ewm_vol_20 = (
    r.ewm(span=20, adjust=False)
    .std()
    * np.sqrt(252)
)

ewm_vol_60 = (
    r.ewm(span=60, adjust=False)
    .std()
    * np.sqrt(252)
)


# Average magnitude of recent returns
mean_abs_return_5 = abs_return.rolling(5).mean()
mean_abs_return_20 = abs_return.rolling(20).mean()


# Largest recent move
max_abs_return_20 = abs_return.rolling(20).max()


# Volatility regime ratios
# > 1 means short-term volatility is above longer-term volatility
rv_ratio_20_60 = rv_20 / rv_60
rv_ratio_60_252 = rv_60 / rv_252


# Volatility of volatility
# Measures how unstable recent volatility itself has been
vol_of_vol_20 = rv_20.rolling(20).std()


# Recent lagged volatility
rv_20_lag1 = rv_20.shift(1)
rv_20_lag5 = rv_20.shift(5)

rv_60_lag1 = rv_60.shift(1)


# Recent return shocks
return_lag1 = r.shift(1)
return_lag2 = r.shift(2)
return_lag5 = r.shift(5)

abs_return_lag1 = abs_return.shift(1)
squared_return_lag1 = squared_return.shift(1)

features = {
    "RV20": rv_20,
    "RV60": rv_60,
    "RV252": rv_252,

    "AbsReturn": abs_return,
    "SquaredReturn": squared_return,

    "EWMVol20": ewm_vol_20,
    "EWMVol60": ewm_vol_60,

    "MeanAbsReturn5": mean_abs_return_5,
    "MeanAbsReturn20": mean_abs_return_20,
    "MaxAbsReturn20": max_abs_return_20,

    "RVRatio20_60": rv_ratio_20_60,
    "RVRatio60_252": rv_ratio_60_252,

    "VolOfVol20": vol_of_vol_20,

    "RV20Lag1": rv_20_lag1,
    "RV20Lag5": rv_20_lag5,
    "RV60Lag1": rv_60_lag1,

    "ReturnLag1": return_lag1,
    "ReturnLag2": return_lag2,
    "ReturnLag5": return_lag5,

    "AbsReturnLag1": abs_return_lag1,
    "SquaredReturnLag1": squared_return_lag1,
}


for name, feature in features.items():

    feature = feature.copy()

    feature.columns = pd.MultiIndex.from_product(
        [[name], feature.columns],
        names=df.columns.names
    )

    df = pd.concat([df, feature], axis=1)

df = df.dropna()

print(df.tail(10))

# --------------------------------------------------
# OPTION CHAINS
# --------------------------------------------------

target_dte = 45

target_date = (
    pd.Timestamp.today().normalize()
    + pd.Timedelta(days=target_dte)
)

chain_calls = {}
chain_puts = {}

for symbol in symbols:

    ticker = yf.Ticker(symbol)

    expiries = ticker.options

    if len(expiries) == 0:
        continue

    # Find available expiry closest to 45 days away
    expiry = min(
        expiries,
        key=lambda x: abs(
            pd.Timestamp(x) - target_date
        )
    )

    chain = ticker.option_chain(expiry)

    call = chain.calls
    put = chain.puts

    call["impliedVolatility"] = call["impliedVolatility"].where(
        call["impliedVolatility"] > 0.000011
    )

    put["impliedVolatility"] = put["impliedVolatility"].where(
        put["impliedVolatility"] > 0.000011
    )

    chain_calls[symbol] = [call, expiry]
    chain_puts[symbol] = [put, expiry]

for ticker in symbols:
    calls = chain_calls[ticker]
    puts = chain_puts[ticker]
    current_price = df[("Close", ticker)].iloc[-1]

    BS_IV_calls, BS_IV_puts = black_scholes(calls,puts,current_price)
    BS_IV_calls.name = "BS_IV"
    BS_IV_puts.name = "BS_IV"

    current_rv20 = df[("RV20", ticker)].iloc[-1]
    BS_RV20_calls, BS_RV20_puts = black_scholes(calls,puts,current_price,sigma_calls=current_rv20)
    BS_RV20_calls.name = "BS_RV20"
    BS_RV20_puts.name = "BS_RV20"

    current_rv60 = df[("RV60", ticker)].iloc[-1]
    BS_RV60_calls, BS_RV60_puts = black_scholes(calls,puts,current_price,sigma_calls=current_rv60)
    BS_RV60_calls.name = "BS_RV60"
    BS_RV60_puts.name = "BS_RV60"

    current_rv252 = df[("RV252", ticker)].iloc[-1]
    BS_RV252_calls, BS_RV252_puts = black_scholes(calls,puts,current_price,sigma_calls=current_rv252)
    BS_RV252_calls.name = "BS_RV252"
    BS_RV252_puts.name = "BS_RV252"

    cols = ["bid", "ask"]
    
    calls[0][cols] = calls[0][cols].replace(0.0, np.nan)

    comparison_calls = calls[0][[
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "impliedVolatility",
        "volume",
        "openInterest"
    ]].copy()

    comparison_calls["Current Stock Price"] = np.ones(len(calls[0])) * current_price

    # Market midpoint
    comparison_calls["MarketMid"] = (
        comparison_calls["bid"] + comparison_calls["ask"]
    ) / 2


    # =========================================================
    # IMPLIED VOLATILITY BLACK-SCHOLES
    # =========================================================

    comparison_calls["IV Used"] = calls[0]["impliedVolatility"]
    comparison_calls["BS_IV"] = BS_IV_calls

    comparison_calls["BS_IV - Mid"] = (
        comparison_calls["BS_IV"]
        - comparison_calls["MarketMid"]
    )

    comparison_calls["BS_IV - Ask"] = (
        comparison_calls["BS_IV"]
        - comparison_calls["ask"]
    )

    comparison_calls["BS_IV - Bid"] = (
        comparison_calls["BS_IV"]
        - comparison_calls["bid"]
    )

    comparison_calls["BS_IV BidEdge"] = (
        comparison_calls["BS_IV"] - comparison_calls["bid"]
    ) / comparison_calls["bid"]

    comparison_calls["BS_IV MidEdge"] = (
        comparison_calls["BS_IV"] - comparison_calls["MarketMid"]
    ) / comparison_calls["MarketMid"]

    comparison_calls["BS_IV AskEdge"] = (
        comparison_calls["BS_IV"] - comparison_calls["ask"]
    ) / comparison_calls["ask"]


    # =========================================================
    # 20-DAY REALISED VOLATILITY BLACK-SCHOLES
    # =========================================================

    comparison_calls["RV20 Used"] = current_rv20
    comparison_calls["BS_RV20"] = BS_RV20_calls

    comparison_calls["BS_RV20 - Mid"] = (
        comparison_calls["BS_RV20"]
        - comparison_calls["MarketMid"]
    )

    comparison_calls["BS_RV20 - Ask"] = (
        comparison_calls["BS_RV20"]
        - comparison_calls["ask"]
    )

    comparison_calls["BS_RV20 - Bid"] = (
        comparison_calls["BS_RV20"]
        - comparison_calls["bid"]
    )

    comparison_calls["BS_RV20 BidEdge"] = (
        comparison_calls["BS_RV20"] - comparison_calls["bid"]
    ) / comparison_calls["bid"]

    comparison_calls["BS_RV20 MidEdge"] = (
        comparison_calls["BS_RV20"] - comparison_calls["MarketMid"]
    ) / comparison_calls["MarketMid"]

    comparison_calls["BS_RV20 AskEdge"] = (
        comparison_calls["BS_RV20"] - comparison_calls["ask"]
    ) / comparison_calls["ask"]


    # =========================================================
    # 60-DAY REALISED VOLATILITY BLACK-SCHOLES
    # =========================================================

    comparison_calls["RV60 Used"] = current_rv60
    comparison_calls["BS_RV60"] = BS_RV60_calls

    comparison_calls["BS_RV60 - Mid"] = (
        comparison_calls["BS_RV60"]
        - comparison_calls["MarketMid"]
    )

    comparison_calls["BS_RV60 - Ask"] = (
        comparison_calls["BS_RV60"]
        - comparison_calls["ask"]
    )

    comparison_calls["BS_RV60 - Bid"] = (
        comparison_calls["BS_RV60"]
        - comparison_calls["bid"]
    )

    comparison_calls["BS_RV60 BidEdge"] = (
        comparison_calls["BS_RV60"] - comparison_calls["bid"]
    ) / comparison_calls["bid"]

    comparison_calls["BS_RV60 MidEdge"] = (
        comparison_calls["BS_RV60"] - comparison_calls["MarketMid"]
    ) / comparison_calls["MarketMid"]

    comparison_calls["BS_RV60 AskEdge"] = (
        comparison_calls["BS_RV60"] - comparison_calls["ask"]
    ) / comparison_calls["ask"]


    # =========================================================
    # 252-DAY REALISED VOLATILITY BLACK-SCHOLES
    # =========================================================

    comparison_calls["RV252 Used"] = current_rv252
    comparison_calls["BS_RV252"] = BS_RV252_calls

    comparison_calls["BS_RV252 - Mid"] = (
        comparison_calls["BS_RV252"]
        - comparison_calls["MarketMid"]
    )

    comparison_calls["BS_RV252 - Ask"] = (
        comparison_calls["BS_RV252"]
        - comparison_calls["ask"]
    )

    comparison_calls["BS_RV252 - Bid"] = (
        comparison_calls["BS_RV252"]
        - comparison_calls["bid"]
    )

    comparison_calls["BS_RV252 BidEdge"] = (
        comparison_calls["BS_RV252"] - comparison_calls["bid"]
    ) / comparison_calls["bid"]

    comparison_calls["BS_RV252 MidEdge"] = (
        comparison_calls["BS_RV252"] - comparison_calls["MarketMid"]
    ) / comparison_calls["MarketMid"]

    comparison_calls["BS_RV252 AskEdge"] = (
        comparison_calls["BS_RV252"] - comparison_calls["ask"]
    ) / comparison_calls["ask"]

    puts[0][cols] = puts[0][cols].replace(0.0, np.nan)

    comparison_puts = puts[0][[
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "impliedVolatility",
        "volume",
        "openInterest"
    ]].copy()

    comparison_puts["Current Stock Price"] = np.ones(len(puts[0])) * current_price

    # Market midpoint
    comparison_puts["MarketMid"] = (
        comparison_puts["bid"] + comparison_puts["ask"]
    ) / 2


    # =========================================================
    # IMPLIED VOLATILITY BLACK-SCHOLES
    # =========================================================

    comparison_puts["IV Used"] = puts[0]["impliedVolatility"]
    comparison_puts["BS_IV"] = BS_IV_puts

    comparison_puts["BS_IV - Mid"] = (
        comparison_puts["BS_IV"]
        - comparison_puts["MarketMid"]
    )

    comparison_puts["BS_IV - Ask"] = (
        comparison_puts["BS_IV"]
        - comparison_puts["ask"]
    )

    comparison_puts["BS_IV - Bid"] = (
        comparison_puts["BS_IV"]
        - comparison_puts["bid"]
    )

    comparison_puts["BS_IV BidEdge"] = (
        comparison_puts["BS_IV"] - comparison_puts["bid"]
    ) / comparison_puts["bid"]

    comparison_puts["BS_IV MidEdge"] = (
        comparison_puts["BS_IV"] - comparison_puts["MarketMid"]
    ) / comparison_puts["MarketMid"]

    comparison_puts["BS_IV AskEdge"] = (
        comparison_puts["BS_IV"] - comparison_puts["ask"]
    ) / comparison_puts["ask"]


    # =========================================================
    # 20-DAY REALISED VOLATILITY BLACK-SCHOLES
    # =========================================================

    comparison_puts["RV20 Used"] = current_rv20
    comparison_puts["BS_RV20"] = BS_RV20_puts

    comparison_puts["BS_RV20 - Mid"] = (
        comparison_puts["BS_RV20"]
        - comparison_puts["MarketMid"]
    )

    comparison_puts["BS_RV20 - Ask"] = (
        comparison_puts["BS_RV20"]
        - comparison_puts["ask"]
    )

    comparison_puts["BS_RV20 - Bid"] = (
        comparison_puts["BS_RV20"]
        - comparison_puts["bid"]
    )

    comparison_puts["BS_RV20 BidEdge"] = (
        comparison_puts["BS_RV20"] - comparison_puts["bid"]
    ) / comparison_puts["bid"]

    comparison_puts["BS_RV20 MidEdge"] = (
        comparison_puts["BS_RV20"] - comparison_puts["MarketMid"]
    ) / comparison_puts["MarketMid"]

    comparison_puts["BS_RV20 AskEdge"] = (
        comparison_puts["BS_RV20"] - comparison_puts["ask"]
    ) / comparison_puts["ask"]


    # =========================================================
    # 60-DAY REALISED VOLATILITY BLACK-SCHOLES
    # =========================================================

    comparison_puts["RV60 Used"] = current_rv60
    comparison_puts["BS_RV60"] = BS_RV60_puts

    comparison_puts["BS_RV60 - Mid"] = (
        comparison_puts["BS_RV60"]
        - comparison_puts["MarketMid"]
    )

    comparison_puts["BS_RV60 - Ask"] = (
        comparison_puts["BS_RV60"]
        - comparison_puts["ask"]
    )

    comparison_puts["BS_RV60 - Bid"] = (
        comparison_puts["BS_RV60"]
        - comparison_puts["bid"]
    )

    comparison_puts["BS_RV60 BidEdge"] = (
        comparison_puts["BS_RV60"] - comparison_puts["bid"]
    ) / comparison_puts["bid"]

    comparison_puts["BS_RV60 MidEdge"] = (
        comparison_puts["BS_RV60"] - comparison_puts["MarketMid"]
    ) / comparison_puts["MarketMid"]

    comparison_puts["BS_RV60 AskEdge"] = (
        comparison_puts["BS_RV60"] - comparison_puts["ask"]
    ) / comparison_puts["ask"]


    # =========================================================
    # 252-DAY REALISED VOLATILITY BLACK-SCHOLES
    # =========================================================

    comparison_puts["RV252 Used"] = current_rv252
    comparison_puts["BS_RV252"] = BS_RV252_puts

    comparison_puts["BS_RV252 - Mid"] = (
        comparison_puts["BS_RV252"]
        - comparison_puts["MarketMid"]
    )

    comparison_puts["BS_RV252 - Ask"] = (
        comparison_puts["BS_RV252"]
        - comparison_puts["ask"]
    )

    comparison_puts["BS_RV252 - Bid"] = (
        comparison_puts["BS_RV252"]
        - comparison_puts["bid"]
    )

    comparison_puts["BS_RV252 BidEdge"] = (
        comparison_puts["BS_RV252"] - comparison_puts["bid"]
    ) / comparison_puts["bid"]

    comparison_puts["BS_RV252 MidEdge"] = (
        comparison_puts["BS_RV252"] - comparison_puts["MarketMid"]
    ) / comparison_puts["MarketMid"]

    comparison_puts["BS_RV252 AskEdge"] = (
        comparison_puts["BS_RV252"] - comparison_puts["ask"]
    ) / comparison_puts["ask"]

    print(comparison_puts)


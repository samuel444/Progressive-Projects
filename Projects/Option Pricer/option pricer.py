import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import norm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def black_scholes(calls,puts,current_price, r=0.0375, sigma_calls = None):
    # Read the expiry dates for the call and put chains
    expiry_calls = calls[1]
    expiry_puts = puts[1]

    # Use market IV unless another volatility value is supplied
    if sigma_calls is None:
        sigma_calls = calls[0]["impliedVolatility"] 
        sigma_puts = puts[0]["impliedVolatility"]
    else:
        sigma_puts = sigma_calls

    # Convert time to expiry into years
    T_calls = (
        pd.Timestamp(expiry_calls) - pd.Timestamp.today().normalize()
    ).days / 365
    T_puts = (
            pd.Timestamp(expiry_puts) - pd.Timestamp.today().normalize()
        ).days / 365
    

    # Get strike prices for each contract
    strike_calls = calls[0]["strike"]
    strike_puts = puts[0]["strike"]

    # Calculate d1 and d2 for calls
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

    # Calculate Black-Scholes call prices
    call_prices = (
        current_price * norm.cdf(d1_calls)
        - strike_calls
        * np.exp(-r * T_calls)
        * norm.cdf(d2_calls)
    )


    # Calculate put prices

    # Calculate d1 and d2 for puts
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

    # Calculate Black-Scholes put prices
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

logger.info("Starting options pricing workflow for %d tickers", len(symbols))

logger.info("Downloading historical stock data")
df = yf.download(
    symbols,
    period="max",
    interval="1d",
    auto_adjust=True,
    progress=False
)

logger.info("Historical stock data downloaded: %d rows", len(df))

# Keep only closing prices
df = df[["Close"]]


# Calculate daily returns

returns = df["Close"].pct_change()

returns.columns = pd.MultiIndex.from_product(
    [["Return"], returns.columns],
    names=df.columns.names
)

df = pd.concat([df, returns], axis=1)
logger.info("Daily returns calculated")


# Get returns with just ticker columns
r = df["Return"]


# Calculate annualised realised volatility
# These values can be used directly as sigma in Black-Scholes

rv_20 = r.rolling(20).std() * np.sqrt(252)
rv_60 = r.rolling(60).std() * np.sqrt(252)
rv_252 = r.rolling(252).std() * np.sqrt(252)
logger.info("RV20, RV60 and RV252 calculated")


# Create features that may help forecast future volatility

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
logger.info("Feature dataset created: %d usable rows", len(df))

# Download option chains

target_dte = 45

target_date = (
    pd.Timestamp.today().normalize()
    + pd.Timedelta(days=target_dte)
)

chain_calls = {}
chain_puts = {}

for symbol in symbols:

    logger.info("Downloading option chain for %s", symbol)
    ticker = yf.Ticker(symbol)

    expiries = ticker.options

    if len(expiries) == 0:
        logger.warning("No option expiries found for %s", symbol)
        continue

    # Find available expiry closest to 45 days away
    expiry = min(
        expiries,
        key=lambda x: abs(
            pd.Timestamp(x) - target_date
        )
    )

    logger.info("%s expiry selected: %s", symbol, expiry)
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
    logger.info(
        "%s option chain loaded: %d calls and %d puts",
        symbol, len(call), len(put)
    )

for ticker in symbols:
    logger.info("Pricing options for %s", ticker)
    calls = chain_calls[ticker]
    puts = chain_puts[ticker]
    current_price = df[("Close", ticker)].iloc[-1]
    logger.info("%s current stock price: %.2f", ticker, current_price)

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

    logger.info(
        "%s volatility inputs - RV20: %.4f, RV60: %.4f, RV252: %.4f",
        ticker, current_rv20, current_rv60, current_rv252
    )
    BS_RV252_calls, BS_RV252_puts = black_scholes(calls,puts,current_price,sigma_calls=current_rv252)
    BS_RV252_calls.name = "BS_RV252"
    BS_RV252_puts.name = "BS_RV252"

    cols = ["bid", "ask"]
    
    # Store calls and puts with their Black-Scholes prices
    option_types = {
        "calls": {
            "chain": calls[0],
            "prices": {
                "IV": BS_IV_calls,
                "RV20": BS_RV20_calls,
                "RV60": BS_RV60_calls,
                "RV252": BS_RV252_calls
            }
        },

        "puts": {
            "chain": puts[0],
            "prices": {
                "IV": BS_IV_puts,
                "RV20": BS_RV20_puts,
                "RV60": BS_RV60_puts,
                "RV252": BS_RV252_puts
            }
        }
    }

    # Store realised volatility values
    volatility_values = {
        "RV20": current_rv20,
        "RV60": current_rv60,
        "RV252": current_rv252
    }

    # Store finished comparison tables
    comparisons = {}

    for option_type, option_data in option_types.items():

        chain = option_data["chain"]
        prices = option_data["prices"]

        # Replace zero bid and ask values because they are not useful market quotes
        chain[cols] = chain[cols].replace(0.0, np.nan)

        logger.info(
            "%s %s with missing bid: %d, missing ask: %d",
            ticker,
            option_type,
            chain["bid"].isna().sum(),
            chain["ask"].isna().sum()
        )

        # Create comparison table
        comparison = chain[[
            "contractSymbol",
            "strike",
            "bid",
            "ask",
            "lastPrice",
            "impliedVolatility",
            "volume",
            "openInterest"
        ]].copy()

        # Add current stock price
        comparison["Current Stock Price"] = current_price

        # Calculate market midpoint
        comparison["MarketMid"] = (
            comparison["bid"] + comparison["ask"]
        ) / 2

        # Loop through each volatility method
        for model, model_prices in prices.items():

            # Add volatility used
            if model == "IV":
                comparison["IV Used"] = chain["impliedVolatility"]

            else:
                comparison[f"{model} Used"] = volatility_values[model]

            # Add Black-Scholes price
            price_column = f"BS_{model}"

            comparison[price_column] = model_prices

            # Compare model price against market prices
            for market_column, market_name in [
                ("MarketMid", "Mid"),
                ("ask", "Ask"),
                ("bid", "Bid")
            ]:

                comparison[f"{price_column} - {market_name}"] = (
                    comparison[price_column]
                    - comparison[market_column]
                )

                comparison[f"{price_column} {market_name}Edge"] = (
                    comparison[price_column]
                    - comparison[market_column]
                ) / comparison[market_column]

        # Save finished table
        comparisons[option_type] = comparison


    comparison_calls = comparisons["calls"]
    comparison_puts = comparisons["puts"]
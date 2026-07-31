import warnings
import logging

import numpy as np
import pandas as pd

from arch import arch_model

from sklearn.decomposition import PCA

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler


ridge_alphas = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1,
    10,
    100,
    1000
]

lasso_alphas = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1
]


har_features = [
    "RV20",
    "RV60",
    "RV252"
]


pca_variance = 0.95

window = 30
step = 30

min_train_rows = 252

# Limit GARCH history for speed
garch_lookback = 1250

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def make_model_specs():

    return [
        ("Train Mean", None),
        ("RV20 Baseline", None),

        ("OLS", None),
        ("OLS PCA", None),

        *[
            ("Ridge", alpha)
            for alpha in ridge_alphas
        ],

        *[
            ("Ridge PCA", alpha)
            for alpha in ridge_alphas
        ],

        *[
            ("Lasso", alpha)
            for alpha in lasso_alphas
        ],

        *[
            ("Lasso PCA", alpha)
            for alpha in lasso_alphas
        ],

        ("Random Forest", None),
        ("Gradient Boosting", None),

        ("HAR-RV", None),
        ("GARCH", None)
    ]

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

def create_model(
    model_name,
    alpha
):

    if model_name in [
        "OLS",
        "OLS PCA",
        "HAR-RV"
    ]:

        return LinearRegression()


    if model_name in [
        "Ridge",
        "Ridge PCA"
    ]:

        return Ridge(
            alpha=alpha
        )


    if model_name in [
        "Lasso",
        "Lasso PCA"
    ]:

        return Lasso(
            alpha=alpha,
            max_iter=100000
        )


    if model_name == "Random Forest":

        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )


    if model_name == "Gradient Boosting":

        return GradientBoostingRegressor(
            random_state=42
        )


    raise ValueError(
        f"Unknown model: {model_name}"
    )

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )


    # Remove invalid predictions if any
    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]


    if len(y_true) == 0:

        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "NRMSE": np.nan,
            "IC": np.nan,
            "N": 0
        }


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    if len(y_true) > 1:

        r2 = r2_score(
            y_true,
            y_pred
        )

    else:

        r2 = np.nan


    target_std = np.std(
        y_true
    )


    nrmse = (
        rmse / target_std
        if target_std > 0
        else np.nan
    )


    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):

        ic = np.corrcoef(
            y_true,
            y_pred
        )[0, 1]

    else:

        ic = np.nan


    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "NRMSE": nrmse,
        "IC": ic,
        "N": len(y_true)
    }

def garch_predictions_for_block(
    ticker_df,
    test_dates,
    rv_length,
    min_returns=252,
    lookback=1250
):

    first_date = test_dates[0]


    # Fit GARCH parameters only on returns
    # before the test block
    fit_returns = (
        ticker_df.loc[
            ticker_df.index < first_date,
            "Return"
        ]
        .dropna()
    )


    if lookback is not None:

        fit_returns = (
            fit_returns
            .iloc[-lookback:]
        )


    if len(fit_returns) < min_returns:

        return None


    # Percentage returns help numerical stability
    fit_returns = (
        fit_returns * 100
    )


    garch = arch_model(
        fit_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False
    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        fitted = garch.fit(
            disp="off",
            show_warning=False
        )


    # Keep parameters fixed during this test block
    params = fitted.params

    predictions = []


    for forecast_date in test_dates:

        # Returns through the forecast date are known
        history = (
            ticker_df
            .loc[
                :forecast_date,
                "Return"
            ]
            .dropna()
        )


        if lookback is not None:

            history = (
                history
                .iloc[-lookback:]
            )


        if len(history) < min_returns:

            predictions.append(
                np.nan
            )

            continue


        history = (
            history * 100
        )


        # Update GARCH state using current history,
        # but do not re-optimise parameters
        state_model = arch_model(
            history,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False
        )


        fixed_model = (
            state_model.fix(
                params
            )
        )


        forecast = (
            fixed_model.forecast(
                horizon=rv_length,
                reindex=False
            )
        )


        future_variance = (
            forecast
            .variance
            .iloc[-1]
            .to_numpy()
        )


        # Convert mean future daily variance
        # into annualised volatility
        predicted_rv = (
            np.sqrt(
                np.mean(
                    future_variance
                )
                * 252
            )
            / 100
        )


        predictions.append(
            predicted_rv
        )


    return np.asarray(
        predictions
    )

def evaluate_symbol_models(
    ticker_df,
    feature_cols,
    rv_length,
    symbol,
    window=30,
    step=30,
    min_train_rows=252
):

    # Only drop rows needed by the models
    required_cols = list(
        dict.fromkeys(
            feature_cols
            + har_features
            + [
                "Return",
                "Target_RV"
            ]
        )
    )


    model_df = (
        ticker_df[
            required_cols
        ]
        .dropna()
        .copy()
    )


    if len(model_df) < (
        min_train_rows
        + rv_length
        + window
    ):

        logger.warning(
            "%s skipped: only %d usable rows",
            symbol,
            len(model_df)
        )

        return pd.DataFrame()


    model_specs = (
        make_model_specs()
    )


    # Store all walk-forward predictions
    prediction_store = {
        spec: {
            "y": [],
            "pred": []
        }

        for spec in model_specs
    }


    # First forecast must have enough training history
    first_test = max(
        int(
            len(model_df)
            * 0.25
        ),

        min_train_rows
        + rv_length
    )


    fold_count = 0


    # Walk-forward folds
    for test_start in range(
        first_test,
        len(model_df) - window + 1,
        step
    ):

        # Purge rv_length rows before test
        train_end = (
            test_start
            - rv_length
        )


        if train_end < min_train_rows:

            continue


        train = model_df.iloc[
            :train_end
        ]


        test = model_df.iloc[
            test_start:
            test_start + window
        ]


        y_train = (
            train["Target_RV"]
            .to_numpy()
        )


        y_test = (
            test["Target_RV"]
            .to_numpy()
        )


        # Raw features
        X_train_raw = (
            train[
                feature_cols
            ]
            .to_numpy()
        )


        X_test_raw = (
            test[
                feature_cols
            ]
            .to_numpy()
        )


        # Scale using this training fold only
        scaler = StandardScaler()


        X_train_scaled = (
            scaler.fit_transform(
                X_train_raw
            )
        )


        X_test_scaled = (
            scaler.transform(
                X_test_raw
            )
        )


        # PCA using this training fold only
        pca = PCA(
            n_components=pca_variance,
            svd_solver="full"
        )


        X_train_pca = (
            pca.fit_transform(
                X_train_scaled
            )
        )


        X_test_pca = (
            pca.transform(
                X_test_scaled
            )
        )


        # HAR features
        X_train_har = (
            train[
                har_features
            ]
            .to_numpy()
        )


        X_test_har = (
            test[
                har_features
            ]
            .to_numpy()
        )


        # Mean baseline
        mean_predictions = np.full(
            len(test),
            y_train.mean()
        )


        prediction_store[
            ("Train Mean", None)
        ]["y"].extend(
            y_test
        )


        prediction_store[
            ("Train Mean", None)
        ]["pred"].extend(
            mean_predictions
        )


        # Simple volatility persistence baseline
        rv20_predictions = (
            test["RV20"]
            .to_numpy()
        )


        prediction_store[
            ("RV20 Baseline", None)
        ]["y"].extend(
            y_test
        )


        prediction_store[
            ("RV20 Baseline", None)
        ]["pred"].extend(
            rv20_predictions
        )


        # Fit supervised models
        for (
            model_name,
            alpha
        ) in model_specs:

            if model_name in [
                "Train Mean",
                "RV20 Baseline",
                "GARCH"
            ]:

                continue


            # Choose correct feature representation
            if model_name == "HAR-RV":

                X_fit = X_train_har
                X_predict = X_test_har


            elif model_name.endswith(
                "PCA"
            ):

                X_fit = X_train_pca
                X_predict = X_test_pca


            elif model_name in [
                "Ridge",
                "Lasso"
            ]:

                X_fit = X_train_scaled
                X_predict = X_test_scaled


            else:

                # OLS and tree models use raw features
                X_fit = X_train_raw
                X_predict = X_test_raw


            current_model = (
                create_model(
                    model_name,
                    alpha
                )
            )


            current_model.fit(
                X_fit,
                y_train
            )


            predictions = (
                current_model.predict(
                    X_predict
                )
            )


            key = (
                model_name,
                alpha
            )


            prediction_store[
                key
            ]["y"].extend(
                y_test
            )


            prediction_store[
                key
            ]["pred"].extend(
                predictions
            )


        # GARCH
        garch_predictions = (
            garch_predictions_for_block(
                ticker_df=ticker_df,
                test_dates=test.index,
                rv_length=rv_length,
                min_returns=min_train_rows,
                lookback=garch_lookback
            )
        )


        if garch_predictions is not None:

            prediction_store[
                ("GARCH", None)
            ]["y"].extend(
                y_test
            )


            prediction_store[
                ("GARCH", None)
            ]["pred"].extend(
                garch_predictions
            )


        fold_count += 1


    logger.info(
        "%s completed %d walk-forward folds",
        symbol,
        fold_count
    )


    # Calculate final metrics
    results = []


    for (
        model_name,
        alpha
    ) in model_specs:

        key = (
            model_name,
            alpha
        )


        metrics = (
            calculate_metrics(
                prediction_store[
                    key
                ]["y"],

                prediction_store[
                    key
                ]["pred"]
            )
        )


        results.append({
            "Model": model_name,
            "Alpha": alpha,
            **metrics
        })


    comparison_table = (
        pd.DataFrame(
            results
        )
        .sort_values(
            by="RMSE",
            na_position="last"
        )
        .reset_index(
            drop=True
        )
    )


    return comparison_table


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

symbols = ["AAPL", "MSFT", "META", "WMT", "GOOGL", "AMZN", "HCA"]

logger.info("Starting options pricing workflow for %d tickers", len(symbols))

logger.info("Downloading historical stock data")
df = yf.download(
    symbols,
    period="5y",
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

target_dte = 45

rv_length = int(45 * (5/7))

target_rv = r.rolling(rv_length).std().shift(-rv_length) * np.sqrt(252)

target_rv.columns = pd.MultiIndex.from_product(
    [["Target_RV"], target_rv.columns],
    names=df.columns.names
)
df = pd.concat([df, target_rv], axis=1)
logger.info("Target RV calculated")

logger.info("Feature dataset created: %d usable rows", len(df))

feature_cols = list(
    features.keys()
)


comparison_tables = {}


for symbol in symbols:

    logger.info(
        "Starting volatility models for %s",
        symbol
    )


    ticker_df = (
        df.xs(
            symbol,
            axis=1,
            level="Ticker"
        )
        .copy()
    )


    comparison_table = (
        evaluate_symbol_models(
            ticker_df=ticker_df,
            feature_cols=feature_cols,
            rv_length=rv_length,
            symbol=symbol,
            window=window,
            step=step,
            min_train_rows=min_train_rows
        )
    )


    comparison_tables[
        symbol
    ] = comparison_table


    print(
        f"\n{symbol}:"
    )

    print(
        comparison_table
    )


# Download option chains
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
    BS_RV252_calls, BS_RV252_puts = black_scholes(calls,puts,current_price,sigma_calls=current_rv252)
    BS_RV252_calls.name = "BS_RV252"
    BS_RV252_puts.name = "BS_RV252"

    current_fv = df[("FV", ticker)].iloc[-1]

    logger.info(
        "%s volatility inputs - RV20: %.4f, RV60: %.4f, RV252: %.4f, FV: %.4f",
        ticker, current_rv20, current_rv60, current_rv252, current_fv
    )
    BS_FV_calls, BS_FV_puts = black_scholes(calls,puts,current_price,sigma_calls=current_fv)
    BS_FV_calls.name = "BS_FV"
    BS_FV_puts.name = "BS_FV"    

    cols = ["bid", "ask"]
    
    # Store calls and puts with their Black-Scholes prices
    option_types = {
        "calls": {
            "chain": calls[0],
            "prices": {
                "IV": BS_IV_calls,
                "RV20": BS_RV20_calls,
                "RV60": BS_RV60_calls,
                "RV252": BS_RV252_calls,
                "FV": BS_FV_calls
            }
        },

        "puts": {
            "chain": puts[0],
            "prices": {
                "IV": BS_IV_puts,
                "RV20": BS_RV20_puts,
                "RV60": BS_RV60_puts,
                "RV252": BS_RV252_puts,
                "FV": BS_FV_puts
            }
        }
    }

    # Store realised volatility values
    volatility_values = {
        "RV20": current_rv20,
        "RV60": current_rv60,
        "RV252": current_rv252,
        "FV": current_fv
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

    comparison_calls["Weighted BS Price"] = ((4*comparison_calls["BS_RV20"]) 
            + (5*comparison_calls["BS_RV60"])
            + comparison_calls["BS_RV252"]
            + (10*comparison_calls["BS_FV"])) / 20
    
    comparison_puts["Weighted BS Price"] = ((4*comparison_puts["BS_RV20"]) 
                + (5*comparison_puts["BS_RV60"])
                + comparison_puts["BS_RV252"]
                + (10*comparison_puts["BS_FV"])) / 20


    # Highlight Significant Contracts Calls
    comparison_calls["SpreadPct"] = (
        comparison_calls["ask"] - comparison_calls["bid"]
    ) / comparison_calls["MarketMid"]

    comparison_calls["Moneyness"] = (
        comparison_calls["strike"]
        / comparison_calls["Current Stock Price"]
    )

    comparison_calls["PositiveRVCount"] = (
        (comparison_calls["BS_RV20 AskEdge"] > 0).astype(int)
        + (comparison_calls["BS_RV60 AskEdge"] > 0).astype(int)
        + (comparison_calls["BS_RV252 AskEdge"] > 0).astype(int)
        + (comparison_calls["BS_FV AskEdge"] > 0).astype(int)
    )

    comparison_calls["WeightedRVAskEdge"] = ((4*comparison_calls["BS_RV20 AskEdge"]) 
            + (5*comparison_calls["BS_RV60 AskEdge"])
            + comparison_calls["BS_RV252 AskEdge"]
            + (10*comparison_calls["BS_FV AskEdge"])) / 20

    highlighted_calls = comparison_calls[
        # Need real executable quotes
        comparison_calls["bid"].notna()
        & comparison_calls["ask"].notna()

        # Avoid extremely wide spreads
        & (comparison_calls["SpreadPct"] <= 0.15)

        # Avoid tiny option prices where percentages become ridiculous
        & (comparison_calls["ask"] >= 0.50)

        # Reasonable liquidity
        & (comparison_calls["openInterest"] >= 100)
        & (comparison_calls["volume"] >= 10)

        # Stay reasonably close to ATM initially
        & (comparison_calls["Moneyness"] >= 0.85)
        & (comparison_calls["Moneyness"] <= 1.15)

        # At least two realised-vol assumptions value it above the ask
        & (comparison_calls["PositiveRVCount"] >= 3)

        # Require a meaningful average/median edge
        & (comparison_calls["WeightedRVAskEdge"] >= 0.05)
    ]



    # Highlight Significant Contracts Puts
    comparison_puts["SpreadPct"] = (
        comparison_puts["ask"] - comparison_puts["bid"]
    ) / comparison_puts["MarketMid"]

    comparison_puts["Moneyness"] = (
        comparison_puts["strike"]
        / comparison_puts["Current Stock Price"]
    )

    comparison_puts["PositiveRVCount"] = (
        (comparison_puts["BS_RV20 AskEdge"] > 0).astype(int)
        + (comparison_puts["BS_RV60 AskEdge"] > 0).astype(int)
        + (comparison_puts["BS_RV252 AskEdge"] > 0).astype(int)
        + (comparison_puts["BS_FV AskEdge"] > 0).astype(int)
    )

    comparison_puts["WeightedRVAskEdge"] = ((4*comparison_puts["BS_RV20 AskEdge"]) 
            + (5*comparison_puts["BS_RV60 AskEdge"])
            + comparison_puts["BS_RV252 AskEdge"]
            + (10*comparison_puts["BS_FV AskEdge"])) / 20

    highlighted_puts = comparison_puts[
        # Need real executable quotes
        comparison_puts["bid"].notna()
        & comparison_puts["ask"].notna()

        # Avoid extremely wide spreads
        & (comparison_puts["SpreadPct"] <= 0.15)

        # Avoid tiny option prices where percentages become ridiculous
        & (comparison_puts["ask"] >= 0.50)

        # Reasonable liquidity
        & (comparison_puts["openInterest"] >= 100)
        & (comparison_puts["volume"] >= 10)

        # Stay reasonably close to ATM initially
        & (comparison_puts["Moneyness"] >= 0.85)
        & (comparison_puts["Moneyness"] <= 1.15)

        # At least two realised-vol assumptions value it above the ask
        & (comparison_puts["PositiveRVCount"] >= 3)

        # Require a meaningful average/median edge
        & (comparison_puts["WeightedRVAskEdge"] >= 0.05)
    ]

    important_cols = [
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "Weighted BS Price",
        "Moneyness",
        "MarketMid",
        "impliedVolatility",
        "RV20 Used",
        "RV60 Used",
        "RV252 Used",
        "BS_RV20 AskEdge",
        "BS_RV60 AskEdge",
        "BS_RV252 AskEdge",
        "BS_FV AskEdge"
        "WeightedRVAskEdge",
        "SpreadPct",
        "volume",
        "openInterest"
    ]

    print(f"{ticker}:")
    print("Highlighted Call Options:")
    print(highlighted_calls[important_cols])

    print(f"\n{ticker}:")
    print("Highlighted Put Options:")
    print(highlighted_puts[important_cols])
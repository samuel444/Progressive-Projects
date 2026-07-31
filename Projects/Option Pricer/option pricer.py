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

from matplotlib import pyplot as plt

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

import yfinance as yf


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

    logger.info(
        "Creating model specifications"
    )

    models = [
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

    logger.info(
        "Created %d model configurations",
        len(models)
    )

    return models


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

    logger.debug(
        "Creating %s model with alpha=%s",
        model_name,
        alpha
    )

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


    logger.error(
        "Unknown model requested: %s",
        model_name
    )

    raise ValueError(
        f"Unknown model: {model_name}"
    )



def calculate_metrics(
    y_true,
    y_pred
):

    logger.debug(
        "Calculating evaluation metrics"
    )

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )


    # Remove invalid predictions
    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    removed = (
        len(y_true)
        - valid.sum()
    )

    if removed > 0:

        logger.warning(
            "Removed %d invalid prediction pairs",
            removed
        )


    y_true = y_true[valid]
    y_pred = y_pred[valid]


    if len(y_true) == 0:

        logger.warning(
            "No valid observations available for metrics"
        )

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


    logger.debug(
        "Metrics complete - N=%d, RMSE=%.5f, MAE=%.5f, R2=%.5f, NRMSE=%.5f, IC=%.5f",
        len(y_true),
        rmse,
        mae,
        r2,
        nrmse,
        ic
    )


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

    logger.debug(
        "Starting GARCH block from %s to %s",
        test_dates[0],
        test_dates[-1]
    )


    first_date = test_dates[0]


    # Returns available before test block
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


    logger.debug(
        "GARCH parameter fit using %d returns",
        len(fit_returns)
    )


    if len(fit_returns) < min_returns:

        logger.warning(
            "Skipping GARCH block: only %d returns, minimum is %d",
            len(fit_returns),
            min_returns
        )

        return None


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


    logger.debug(
        "Optimising GARCH(1,1) parameters"
    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        fitted = garch.fit(
            disp="off",
            show_warning=False
        )


    params = fitted.params


    logger.debug(
        "GARCH parameter fit complete"
    )


    predictions = []


    for forecast_date in test_dates:

        history = (
            ticker_df.loc[
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


    predictions = np.asarray(
        predictions
    )


    logger.debug(
        "GARCH block complete with %d predictions",
        np.isfinite(predictions).sum()
    )


    return predictions


def evaluate_symbol_models(
    ticker_df,
    feature_cols,
    rv_length,
    symbol,
    window=30,
    step=30,
    min_train_rows=252
):

    logger.info(
        "%s starting model evaluation",
        symbol
    )


    # Select required data
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


    logger.info(
        "%s has %d usable rows and %d features",
        symbol,
        len(model_df),
        len(feature_cols)
    )


    minimum_required = (
        min_train_rows
        + rv_length
        + window
    )


    if len(model_df) < minimum_required:

        logger.warning(
            "%s skipped: %d rows available, %d required",
            symbol,
            len(model_df),
            minimum_required
        )

        return pd.DataFrame()


    model_specs = (
        make_model_specs()
    )


    prediction_store = {
        spec: {
            "y": [],
            "pred": []
        }

        for spec in model_specs
    }


    first_test = max(
        int(
            len(model_df)
            * 0.25
        ),

        min_train_rows
        + rv_length
    )


    logger.info(
        "%s walk-forward starts at row %d with window=%d, step=%d, purge=%d",
        symbol,
        first_test,
        window,
        step,
        rv_length
    )


    fold_count = 0


    for test_start in range(
        first_test,
        len(model_df) - window + 1,
        step
    ):

        train_end = (
            test_start
            - rv_length
        )


        if train_end < min_train_rows:

            logger.debug(
                "%s fold skipped: training rows=%d",
                symbol,
                train_end
            )

            continue


        train = model_df.iloc[
            :train_end
        ]


        test = model_df.iloc[
            test_start:
            test_start + window
        ]


        fold_count += 1


        logger.debug(
            "%s fold %d - train=%d rows, purge=%d, test=%d rows, test start=%s",
            symbol,
            fold_count,
            len(train),
            rv_length,
            len(test),
            test.index[0]
        )


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


        # Scale inside fold
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


        logger.debug(
            "%s fold %d scaling complete",
            symbol,
            fold_count
        )


        # Fit PCA inside fold
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


        logger.debug(
            "%s fold %d PCA retained %d of %d components",
            symbol,
            fold_count,
            X_train_pca.shape[1],
            X_train_raw.shape[1]
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


        # RV20 baseline
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


        logger.debug(
            "%s fold %d baselines complete",
            symbol,
            fold_count
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


        logger.debug(
            "%s fold %d supervised models complete",
            symbol,
            fold_count
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


        logger.debug(
            "%s fold %d complete",
            symbol,
            fold_count
        )


    logger.info(
        "%s completed %d walk-forward folds",
        symbol,
        fold_count
    )


    # Calculate model metrics
    logger.info(
        "%s calculating model comparison metrics",
        symbol
    )


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


        logger.info(
            "%s %-20s alpha=%s RMSE=%.5f NRMSE=%.5f R2=%.5f IC=%.5f",
            symbol,
            model_name,
            alpha,
            metrics["RMSE"],
            metrics["NRMSE"],
            metrics["R2"],
            metrics["IC"]
        )


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


    if not comparison_table.empty:

        best = (
            comparison_table
            .iloc[0]
        )


        logger.info(
            "%s evaluation complete - best model=%s, alpha=%s, RMSE=%.5f, NRMSE=%.5f",
            symbol,
            best["Model"],
            best["Alpha"],
            best["RMSE"],
            best["NRMSE"]
        )


    return comparison_table

def monte_carlo_option_chain(
    ticker,
    calls,
    puts,
    df,
    r=0.0375,
    q=0.0,
    simulations=10000,
    vol_lookback=1260
):

    logger.info(
        "%s starting Monte Carlo simulation",
        ticker
    )

    call_chain = calls[0].copy()
    put_chain = puts[0].copy()

    expiry = pd.Timestamp(
        calls[1]
    )

    today = pd.Timestamp.today().normalize()


    # Time until the actual option expiry
    calendar_days = (
        expiry - today
    ).days

    if calendar_days <= 0:

        raise ValueError(
            f"{ticker} option expiry has passed"
        )


    T = calendar_days / 365


    # Number of trading steps until expiry
    steps = np.busday_count(
        np.datetime64(today.date()),
        np.datetime64(expiry.date())
    )

    steps = max(
        int(steps),
        1
    )

    dt = T / steps


    # Current stock price
    S0 = float(
        df[("Close", ticker)]
        .dropna()
        .iloc[-1]
    )


    # Train-mean volatility forecast
    historical_targets = (
        df[("Target_RV", ticker)]
        .dropna()
        .iloc[-vol_lookback:]
    )

    sigma = float(
        historical_targets.mean()
    )


    if (
        not np.isfinite(sigma)
        or sigma <= 0
    ):

        raise ValueError(
            f"{ticker} has an invalid volatility forecast"
        )


    logger.info(
        "%s Monte Carlo inputs - S0: %.2f, sigma: %.4f, T: %.4f, steps: %d, simulations: %d",
        ticker,
        S0,
        sigma,
        T,
        steps,
        simulations
    )


    rng = np.random.default_rng()


    # Antithetic samples reduce simulation noise
    half_simulations = (
        simulations + 1
    ) // 2

    z_half = rng.standard_normal(
        size=(
            steps,
            half_simulations
        )
    )

    z = np.concatenate(
        [
            z_half,
            -z_half
        ],
        axis=1
    )

    z = z[
        :,
        :simulations
    ]


    # Risk-neutral GBM increments
    log_increments = (
        (
            r
            - q
            - 0.5 * sigma**2
        )
        * dt
        + sigma
        * np.sqrt(dt)
        * z
    )


    # Build complete simulated paths
    cumulative_log_returns = np.cumsum(
        log_increments,
        axis=0
    )

    cumulative_log_returns = np.vstack(
        [
            np.zeros(
                simulations
            ),
            cumulative_log_returns
        ]
    )

    paths = (
        S0
        * np.exp(
            cumulative_log_returns
        )
    )


    terminal_prices = paths[-1]

    discount_factor = np.exp(
        -r * T
    )


    # Price every call strike
    call_strikes = (
        call_chain["strike"]
        .to_numpy(
            dtype=float
        )
    )

    call_payoffs = np.maximum(
        terminal_prices[:, None]
        - call_strikes[None, :],
        0
    )

    call_prices = (
        discount_factor
        * call_payoffs.mean(
            axis=0
        )
    )

    call_standard_errors = (
        discount_factor
        * call_payoffs.std(
            axis=0,
            ddof=1
        )
        / np.sqrt(simulations)
    )


    # Price every put strike
    put_strikes = (
        put_chain["strike"]
        .to_numpy(
            dtype=float
        )
    )

    put_payoffs = np.maximum(
        put_strikes[None, :]
        - terminal_prices[:, None],
        0
    )

    put_prices = (
        discount_factor
        * put_payoffs.mean(
            axis=0
        )
    )

    put_standard_errors = (
        discount_factor
        * put_payoffs.std(
            axis=0,
            ddof=1
        )
        / np.sqrt(simulations)
    )


    # Add results to option chains
    call_chain["MC_FV"] = (
        call_prices
    )

    call_chain["MC_SE"] = (
        call_standard_errors
    )

    call_chain["MC AskEdge"] = (
        call_chain["MC_FV"]
        - call_chain["ask"]
    ) / call_chain["ask"]


    put_chain["MC_FV"] = (
        put_prices
    )

    put_chain["MC_SE"] = (
        put_standard_errors
    )

    put_chain["MC AskEdge"] = (
        put_chain["MC_FV"]
        - put_chain["ask"]
    ) / put_chain["ask"]


    logger.info(
        "%s Monte Carlo simulation complete",
        ticker
    )


    return {
        "paths": paths,
        "terminal_prices": terminal_prices,
        "calls": call_chain,
        "puts": put_chain,
        "sigma": sigma,
        "T": T,
        "steps": steps
    }


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

'''feature_cols = list(
    features.keys()
)


comparison_tables = {}


logger.info(
    "Starting volatility forecasting for %d symbols",
    len(symbols)
)


for symbol in symbols:

    logger.info(
        "Starting %s",
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


    logger.info(
        "%s extracted with %d rows",
        symbol,
        len(ticker_df)
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


    logger.info(
        "%s finished",
        symbol
    )


    print(
        f"\n{symbol}:"
    )

    print(
        comparison_table
    )'''


logger.info(
    "Volatility forecasting complete for all symbols"
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

    current_fv = df[("Target_RV", ticker)].iloc[-1260:].mean()

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

    # Minimum edge required for a buy signal
    buy_edge = 0.05


    comparison_calls["Recommended Action"] = np.select(
        [
            comparison_calls["BS_FV AskEdge"].isna(),

            comparison_calls["BS_FV AskEdge"] >= buy_edge,

            (
                comparison_calls["BS_FV AskEdge"] > 0
            )
            & (
                comparison_calls["BS_FV AskEdge"] < buy_edge
            ),

            comparison_calls["BS_FV AskEdge"] <= 0
        ],
        [
            "No Data",
            "Buy",
            "Positive Edge",
            "Do Not Buy"
        ],
        default="No Data"
    )

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
        & (comparison_calls["BS_FV AskEdge"] >= 0.05)
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

# Minimum edge required for a buy signal
    buy_edge = 0.05


    comparison_puts["Recommended Action"] = np.select(
        [
            comparison_puts["BS_FV AskEdge"].isna(),

            comparison_puts["BS_FV AskEdge"] >= buy_edge,

            (
                comparison_puts["BS_FV AskEdge"] > 0
            )
            & (
                comparison_puts["BS_FV AskEdge"] < buy_edge
            ),

            comparison_puts["BS_FV AskEdge"] <= 0
        ],
        [
            "No Data",
            "Buy",
            "Positive Edge",
            "Do Not Buy"
        ],
        default="No Data"
    )

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
        & (comparison_puts["BS_FV AskEdge"] >= 0.05)
    ]

    important_cols = [
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "BS_FV",
        "BS_FV AskEdge",
        "Moneyness",
        "MarketMid",
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


monte_carlo_results = {}


for ticker in symbols:

    if (
        ticker not in chain_calls
        or ticker not in chain_puts
    ):

        logger.warning(
            "%s skipped because option chain is unavailable",
            ticker
        )

        continue


    result = monte_carlo_option_chain(
        ticker=ticker,
        calls=chain_calls[ticker],
        puts=chain_puts[ticker],
        df=df,
        r=0.0375,
        simulations=10000,
        vol_lookback=1260
    )


    monte_carlo_results[
        ticker
    ] = result


    paths = result["paths"]
    terminal_prices = (
        result["terminal_prices"]
    )

    median_path = np.median(paths, axis=1)


    # Plot the first 100 paths
    plt.plot(
        paths[:, :100], alpha=0.4
    )

    plt.plot(
        median_path,
        linewidth=3,
        label="Median simulation"
    )

    plt.xlabel(
        "Trading Days"
    )

    plt.ylabel(
        "Stock Price ($)"
    )

    plt.title(
        f"{ticker} Monte Carlo Simulation"
    )

    plt.show()


    # Plot terminal price distribution
    plt.hist(
        terminal_prices,
        bins=50
    )

    plt.axvline(
        df[("Close", ticker)]
        .dropna()
        .iloc[-1],
        linestyle="--",
        label="Current Price"
    )

    plt.xlabel(
        "Price at Expiry ($)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{ticker} Price Distribution at Expiry"
    )

    plt.legend()

    plt.show()
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
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

from matplotlib import pyplot as plt, ticker

from scipy.stats import norm

from scipy.optimize import brentq

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

GREEK_VALIDATION_TOLERANCES = {
    "Delta": {
        "atol": 1e-5,
        "rtol": 1e-4,
    },
    "Gamma": {
        "atol": 1e-6,
        "rtol": 1e-3,
    },
    "Vega": {
        "atol": 1e-5,
        "rtol": 1e-3,
    },
    "Theta": {
        "atol": 1e-5,
        "rtol": 1e-3,
    },
    "Rho": {
        "atol": 1e-5,
        "rtol": 1e-3,
    },
}


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


@dataclass
class OptionTicker:
    """Store all market, forecast and option-chain state for one ticker.

    The object is deliberately a state holder rather than a pricing engine.
    Pricing, validation and simulation remain in separate functions so they
    can be tested independently.
    """

    symbol: str
    target_dte: int = 45
    risk_free_rate: float = 0.0375
    dividend_yield: float = 0.0
    forecast_lookback: int = 1260
    buy_edge: float = 0.05
    parity_tolerance: float = 1e-6
    valuation_date: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp.today().normalize()
    )

    forecast_horizon: int = field(init=False)
    current_price: Optional[float] = None
    expiry: Optional[pd.Timestamp] = None
    calendar_dte: Optional[int] = None
    trading_dte: Optional[int] = None
    time_to_expiry: Optional[float] = None
    forward_price: Optional[float] = None
    discounted_spot: Optional[float] = None

    volatility_inputs: dict[str, float] = field(default_factory=dict)
    call_chain: Optional[pd.DataFrame] = field(default=None, repr=False)
    put_chain: Optional[pd.DataFrame] = field(default=None, repr=False)

    call_analysis: Optional[pd.DataFrame] = field(default=None, repr=False)
    put_analysis: Optional[pd.DataFrame] = field(default=None, repr=False)
    parity_table: Optional[pd.DataFrame] = field(default=None, repr=False)
    monte_carlo_result: Optional[dict[str, Any]] = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.valuation_date = pd.Timestamp(self.valuation_date).normalize()

        if not self.symbol:
            raise ValueError("Ticker symbol cannot be empty")
        if self.target_dte <= 0:
            raise ValueError("target_dte must be positive")
        if self.risk_free_rate <= -1:
            raise ValueError("risk_free_rate is invalid")
        if self.dividend_yield < 0:
            raise ValueError("dividend_yield cannot be negative")
        if self.forecast_lookback <= 0:
            raise ValueError("forecast_lookback must be positive")

        self.forecast_horizon = self._trading_days_to_target()

    def _trading_days_to_target(self) -> int:
        target_date = (
            self.valuation_date
            + pd.Timedelta(days=self.target_dte)
        )

        return max(
            int(
                np.busday_count(
                    self.valuation_date.date(),
                    target_date.date(),
                )
            ),
            1,
        )

    @property
    def target_date(self) -> pd.Timestamp:
        return (
            self.valuation_date
            + pd.Timedelta(days=self.target_dte)
        )

    @property
    def expiry_string(self) -> str:
        if self.expiry is None:
            raise ValueError(f"{self.symbol} does not have an expiry")
        return self.expiry.strftime("%Y-%m-%d")

    @property
    def call_record(self) -> tuple[pd.DataFrame, str]:
        if self.call_chain is None:
            raise ValueError(f"{self.symbol} does not have a call chain")
        return self.call_chain, self.expiry_string

    @property
    def put_record(self) -> tuple[pd.DataFrame, str]:
        if self.put_chain is None:
            raise ValueError(f"{self.symbol} does not have a put chain")
        return self.put_chain, self.expiry_string

    def attach_option_chains(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        expiry: str,
    ) -> None:
        if not isinstance(calls, pd.DataFrame) or calls.empty:
            raise ValueError(f"{self.symbol} call chain is empty or invalid")
        if not isinstance(puts, pd.DataFrame) or puts.empty:
            raise ValueError(f"{self.symbol} put chain is empty or invalid")

        self.call_chain = calls.copy()
        self.put_chain = puts.copy()
        self.expiry = pd.Timestamp(expiry).normalize()
        self.refresh_derived_market_values()

    def set_current_price(self, current_price: float) -> None:
        current_price = float(current_price)
        if not np.isfinite(current_price) or current_price <= 0:
            raise ValueError(
                f"{self.symbol} current price must be finite and positive"
            )

        self.current_price = current_price
        self.refresh_derived_market_values()

    def set_volatility_inputs(
        self,
        volatility_inputs: Mapping[str, float],
    ) -> None:
        converted = {
            name: float(value)
            for name, value in volatility_inputs.items()
        }

        invalid = {
            name: value
            for name, value in converted.items()
            if not np.isfinite(value) or value <= 0
        }

        if invalid:
            raise ValueError(
                f"Invalid volatility inputs for {self.symbol}: {invalid}"
            )

        self.volatility_inputs = converted

    def refresh_derived_market_values(self) -> None:
        if self.expiry is not None:
            self.calendar_dte = max(
                (self.expiry - self.valuation_date).days,
                0,
            )
            self.trading_dte = max(
                int(
                    np.busday_count(
                        self.valuation_date.date(),
                        self.expiry.date(),
                    )
                ),
                0,
            )
            self.time_to_expiry = self.calendar_dte / 365.0

        if (
            self.current_price is not None
            and self.time_to_expiry is not None
        ):
            self.discounted_spot = (
                self.current_price
                * np.exp(-self.dividend_yield * self.time_to_expiry)
            )
            self.forward_price = (
                self.current_price
                * np.exp(
                    (
                        self.risk_free_rate
                        - self.dividend_yield
                    )
                    * self.time_to_expiry
                )
            )

    def ensure_pricing_ready(self) -> None:
        missing = []

        if self.call_chain is None:
            missing.append("call_chain")
        if self.put_chain is None:
            missing.append("put_chain")
        if self.expiry is None:
            missing.append("expiry")
        if self.current_price is None:
            missing.append("current_price")
        if self.time_to_expiry is None:
            missing.append("time_to_expiry")

        if missing:
            raise ValueError(
                f"{self.symbol} is not ready for pricing; missing {missing}"
            )

    def table_key(self, option_type: str) -> str:
        return f"{self.symbol} {option_type}"


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



def black_scholes(
    ticker,
    spot,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield,
    option_type,
    volatility: int = None,
):
    """Price the ticker's European call and put chains consistently."""

    d1 = (
        np.log(spot / strike)
        + (
            risk_free_rate
            - dividend_yield
            + volatility**2 / 2
        ) * time_to_expiry
    ) / (
        volatility * np.sqrt(time_to_expiry)
    )

    d2 = (
        d1
        - volatility * np.sqrt(time_to_expiry)
    )

    if option_type == "call":

        volatility = ticker.call_chain["impliedVolatility"] if volatility is None else volatility
        
        with np.errstate(divide="ignore", invalid="ignore"):

            call_prices = (
                spot
                * np.exp(-dividend_yield * time_to_expiry)
                * norm.cdf(d1)
                - strike
                * np.exp(-risk_free_rate * time_to_expiry)
                * norm.cdf(d2)
            )

        return call_prices
    
    elif option_type == "put":

        volatility = (
            ticker.put_chain["impliedVolatility"]
            if volatility is None
            else volatility
        )

        with np.errstate(divide="ignore", invalid="ignore"):

            prices = (
                strike
                * np.exp(-risk_free_rate * time_to_expiry)
                * norm.cdf(-d2)
                - spot
                * np.exp(-dividend_yield * time_to_expiry)
                * norm.cdf(-d1)
            )

        return prices
    
    else:

        raise ValueError(
            f"Invalid option_type: {option_type}. Must be 'call' or 'put'."
        )

    


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
    ticker: OptionTicker,
    window=30,
    step=30,
    min_train_rows=252
):
    symbol = ticker.symbol
    rv_length = ticker.forecast_horizon

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


def evaluate_historical_mean_holdout(
    ticker_df,
    ticker: OptionTicker,
    test_fraction=0.20
):
    symbol = ticker.symbol
    rv_length = ticker.forecast_horizon
    """
    Evaluate a constant historical-mean volatility forecast using
    one chronological train/test split.
    """

    target = (
        ticker_df["Target_RV"]
        .dropna()
        .astype(float)
    )

    split_index = int(
        len(target) * (1 - test_fraction)
    )

    # Purge observations whose future volatility window
    # overlaps the test period
    train_end = split_index - rv_length

    if train_end <= 0:
        raise ValueError(
            f"{symbol} does not have enough observations "
            "for the requested train/test split and purge."
        )

    y_train = (
        target
        .iloc[:train_end]
        .to_numpy()
    )

    y_test = (
        target
        .iloc[split_index:]
        .to_numpy()
    )

    historical_mean = float(
        y_train.mean()
    )

    predictions = np.full(
        shape=len(y_test),
        fill_value=historical_mean,
        dtype=float
    )

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=predictions
    )

    return pd.DataFrame([
        {
            "Symbol": symbol,
            "Model": "Historical Mean Holdout",
            "Forecast": historical_mean,
            "Train Rows": len(y_train),
            "Purged Rows": rv_length,
            "Test Rows": len(y_test),
            **metrics
        }
    ])



def monte_carlo_option_chain(
    ticker: OptionTicker,
    simulations: int = 10000,
    random_seed: Optional[int] = None,
):
    """Price one ticker's chains using the same stored market assumptions."""

    ticker.ensure_pricing_ready()

    if "ForeV" not in ticker.volatility_inputs:
        raise ValueError(
            f"{ticker.symbol} does not have a ForeV volatility input"
        )

    logger.info(
        "%s starting Monte Carlo simulation",
        ticker.symbol,
    )

    call_chain = ticker.call_chain.copy()
    put_chain = ticker.put_chain.copy()

    time_to_expiry = ticker.time_to_expiry
    steps = max(int(ticker.trading_dte), 1)
    dt = time_to_expiry / steps

    current_price = ticker.current_price
    sigma = ticker.volatility_inputs["ForeV"]
    risk_free_rate = ticker.risk_free_rate
    dividend_yield = ticker.dividend_yield

    logger.info(
        "%s Monte Carlo inputs - S0: %.2f, sigma: %.4f, "
        "T: %.4f, steps: %d, simulations: %d",
        ticker.symbol,
        current_price,
        sigma,
        time_to_expiry,
        steps,
        simulations,
    )

    rng = np.random.default_rng(random_seed)

    half_simulations = (simulations + 1) // 2
    z_half = rng.standard_normal(
        size=(steps, half_simulations)
    )
    z = np.concatenate([z_half, -z_half], axis=1)
    z = z[:, :simulations]

    log_increments = (
        (
            risk_free_rate
            - dividend_yield
            - 0.5 * sigma**2
        )
        * dt
        + sigma * np.sqrt(dt) * z
    )

    cumulative_log_returns = np.cumsum(
        log_increments,
        axis=0,
    )
    cumulative_log_returns = np.vstack(
        [
            np.zeros(simulations),
            cumulative_log_returns,
        ]
    )

    paths = current_price * np.exp(cumulative_log_returns)
    terminal_prices = paths[-1]
    discount_factor = np.exp(
        -risk_free_rate * time_to_expiry
    )

    call_strikes = call_chain["strike"].to_numpy(dtype=float)
    call_payoffs = np.maximum(
        terminal_prices[:, None] - call_strikes[None, :],
        0.0,
    )
    call_prices = discount_factor * call_payoffs.mean(axis=0)
    call_standard_errors = (
        discount_factor
        * call_payoffs.std(axis=0, ddof=1)
        / np.sqrt(simulations)
    )

    put_strikes = put_chain["strike"].to_numpy(dtype=float)
    put_payoffs = np.maximum(
        put_strikes[None, :] - terminal_prices[:, None],
        0.0,
    )
    put_prices = discount_factor * put_payoffs.mean(axis=0)
    put_standard_errors = (
        discount_factor
        * put_payoffs.std(axis=0, ddof=1)
        / np.sqrt(simulations)
    )

    call_chain["MC_ForeV"] = call_prices
    call_chain["MC_SE"] = call_standard_errors
    call_chain["MC AskEdge"] = safe_relative_edge(
        call_chain["MC_ForeV"],
        call_chain["ask"],
    )

    put_chain["MC_ForeV"] = put_prices
    put_chain["MC_SE"] = put_standard_errors
    put_chain["MC AskEdge"] = safe_relative_edge(
        put_chain["MC_ForeV"],
        put_chain["ask"],
    )

    logger.info(
        "%s Monte Carlo simulation complete",
        ticker.symbol,
    )

    result = {
        "paths": paths,
        "terminal_prices": terminal_prices,
        "calls": call_chain,
        "puts": put_chain,
        "sigma": sigma,
        "T": time_to_expiry,
        "steps": steps,
    }
    ticker.monte_carlo_result = result
    return result



# Pricing and filtering configuration
RISK_FREE_RATE = 0.0375
FORECAST_VOL_LOOKBACK = 1260
BUY_EDGE = 0.05
PARITY_TOLERANCE = 1e-6
MONTE_CARLO_SIMULATIONS = 10_000


# Market-quality filters
MAX_SPREAD_PCT = 0.15
MIN_ASK = 0.50
MIN_OPEN_INTEREST = 100
MIN_VOLUME = 10
MIN_MONEYNESS = 0.85
MAX_MONEYNESS = 1.15

# Plotting can be disabled when running the engine repeatedly
PLOT_MONTE_CARLO = True


OptionChainRecord = tuple[pd.DataFrame, str]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def years_to_expiry(
    expiry: str,
    valuation_date: pd.Timestamp,
) -> float:
    """Return calendar time to expiry in years using an ACT/365 convention."""

    if valuation_date is None:
        valuation_date = pd.Timestamp.today().normalize()
    else:
        valuation_date = pd.Timestamp(valuation_date).normalize()

    expiry_date = pd.Timestamp(expiry).normalize()
    calendar_days = max((expiry_date - valuation_date).days, 0)

    return calendar_days / 365.0



def latest_feature_value(
    data: pd.DataFrame,
    feature: str,
    ticker: OptionTicker,
) -> float:
    """Return the latest non-missing feature value for one ticker object."""

    column = (feature, ticker.symbol)

    if column not in data.columns:
        raise KeyError(f"Missing column {column!r}")

    values = data[column].dropna()

    if values.empty:
        raise ValueError(
            f"No usable values found for {feature} and {ticker.symbol}"
        )

    value = float(values.iloc[-1])

    if not np.isfinite(value):
        raise ValueError(
            f"Latest {feature} value for {ticker.symbol} is not finite"
        )

    return value



def safe_relative_edge(
    model_value: pd.Series,
    market_value: pd.Series,
) -> pd.Series:
    """Calculate (model - market) / market without creating infinities."""

    model_value = pd.to_numeric(model_value, errors="coerce")
    market_value = pd.to_numeric(market_value, errors="coerce")

    valid_market = market_value > 0

    edge = pd.Series(
        np.nan,
        index=model_value.index,
        dtype=float,
    )

    edge.loc[valid_market] = (
        model_value.loc[valid_market]
        - market_value.loc[valid_market]
    ) / market_value.loc[valid_market]

    return edge


# ---------------------------------------------------------------------------
# Option-chain download
# ---------------------------------------------------------------------------


def choose_closest_expiry(
    expiries: list[str],
    target_date: pd.Timestamp,
) -> str:
    """Choose the listed expiry closest to the requested target date."""

    expiry_list = list(expiries)

    if not expiry_list:
        raise ValueError("No option expiries were supplied")

    return min(
        expiry_list,
        key=lambda expiry: abs(
            pd.Timestamp(expiry).normalize()
            - target_date.normalize()
        ),
    )


def clean_downloaded_chain(
    chain: pd.DataFrame,
    iv_floor: float = 0.000011,
) -> pd.DataFrame:
    """Copy an option chain and replace unusable quotes and IV values."""

    cleaned = chain.copy()

    if "impliedVolatility" in cleaned.columns:
        cleaned["impliedVolatility"] = cleaned[
            "impliedVolatility"
        ].where(
            cleaned["impliedVolatility"] > iv_floor
        )

    quote_columns = [
        column
        for column in ("bid", "ask")
        if column in cleaned.columns
    ]

    if quote_columns:
        cleaned[quote_columns] = cleaned[quote_columns].replace(
            0.0,
            np.nan,
        )

    return cleaned



def download_option_chains(
    tickers: list[OptionTicker],
) -> list[OptionTicker]:
    """Download and attach the nearest target-DTE chain to each ticker."""

    logger.info(
        "Downloading option chains for %d ticker objects",
        len(tickers),
    )

    loaded = 0

    for ticker in tickers:
        logger.info(
            "Downloading option chain for %s nearest to %s",
            ticker.symbol,
            ticker.target_date.date(),
        )

        try:
            yf_ticker = yf.Ticker(ticker.symbol)
            expiries = yf_ticker.options

            if not expiries:
                logger.warning(
                    "No option expiries found for %s",
                    ticker.symbol,
                )
                continue

            expiry = choose_closest_expiry(
                expiries=expiries,
                target_date=ticker.target_date,
            )
            downloaded_chain = yf_ticker.option_chain(expiry)

            calls = clean_downloaded_chain(downloaded_chain.calls)
            puts = clean_downloaded_chain(downloaded_chain.puts)

            ticker.attach_option_chains(
                calls=calls,
                puts=puts,
                expiry=expiry,
            )
            loaded += 1

            logger.info(
                "%s chain loaded for %s: %d calls and %d puts",
                ticker.symbol,
                ticker.expiry_string,
                len(calls),
                len(puts),
            )

        except Exception:
            logger.exception(
                "Failed to download option chain for %s",
                ticker.symbol,
            )

    logger.info(
        "Option-chain download complete: %d of %d tickers loaded",
        loaded,
        len(tickers),
    )

    return tickers



# ---------------------------------------------------------------------------
# Volatility inputs and Black-Scholes pricing
# ---------------------------------------------------------------------------



def get_volatility_inputs(
    data: pd.DataFrame,
    ticker: OptionTicker,
) -> dict[str, float]:
    """Collect and store realised and forecast volatility inputs."""

    target_values = (
        data[("Target_RV", ticker.symbol)]
        .dropna()
        .tail(ticker.forecast_lookback)
    )

    if target_values.empty:
        raise ValueError(
            f"No Target_RV values available for {ticker.symbol}"
        )

    volatility_inputs = {
        "RV20": latest_feature_value(data, "RV20", ticker),
        "RV60": latest_feature_value(data, "RV60", ticker),
        "RV252": latest_feature_value(data, "RV252", ticker),
        "ForeV": float(target_values.mean()),
    }

    ticker.set_volatility_inputs(volatility_inputs)

    logger.info(
        "%s volatility inputs - RV20: %.4f, RV60: %.4f, "
        "RV252: %.4f, forecast: %.4f",
        ticker.symbol,
        ticker.volatility_inputs["RV20"],
        ticker.volatility_inputs["RV60"],
        ticker.volatility_inputs["RV252"],
        ticker.volatility_inputs["ForeV"],
    )

    return ticker.volatility_inputs



def as_aligned_series(
    values: Any,
    index: pd.Index,
    name: str,
) -> pd.Series:
    """Convert model output to a Series aligned to an option-chain index."""

    array = np.asarray(values, dtype=float)

    if array.ndim != 1 or len(array) != len(index):
        raise ValueError(
            f"{name} has shape {array.shape}; expected ({len(index)},)"
        )

    return pd.Series(array, index=index, name=name)



def calculate_black_scholes_scenarios(
    ticker: OptionTicker,
) -> dict[str, dict[str, pd.Series]]:
    """Price calls and puts under the ticker object's volatility inputs."""

    ticker.ensure_pricing_ready()

    scenario_prices: dict[str, dict[str, pd.Series]] = {
        "calls": {},
        "puts": {},
    }

    iv_call_prices = black_scholes(
        ticker=ticker,
        spot=ticker.current_price,
        strike=ticker.call_chain["strike"],
        time_to_expiry=ticker.time_to_expiry,
        risk_free_rate=ticker.risk_free_rate,
        dividend_yield=ticker.dividend_yield,
        volatility=ticker.call_chain["impliedVolatility"],
        option_type="call",
    )

    iv_put_prices = black_scholes(
        ticker=ticker,
        spot=ticker.current_price,
        strike=ticker.put_chain["strike"],
        time_to_expiry=ticker.time_to_expiry,
        risk_free_rate=ticker.risk_free_rate,
        dividend_yield=ticker.dividend_yield,
        volatility=ticker.put_chain["impliedVolatility"],
        option_type="put",
    )

    scenario_prices["calls"]["IV"] = as_aligned_series(
        iv_call_prices,
        ticker.call_chain.index,
        "BS_IV",
    )
    scenario_prices["puts"]["IV"] = as_aligned_series(
        iv_put_prices,
        ticker.put_chain.index,
        "BS_IV",
    )

    for model_name, sigma in ticker.volatility_inputs.items():
        model_call_prices = black_scholes(
            ticker=ticker,
            spot=ticker.current_price,
            strike=ticker.call_chain["strike"],
            time_to_expiry=ticker.time_to_expiry,
            risk_free_rate=ticker.risk_free_rate,
            dividend_yield=ticker.dividend_yield,
            volatility=sigma,
            option_type="call"
        )

        model_put_prices = black_scholes(
            ticker=ticker,
            spot=ticker.current_price,
            strike=ticker.put_chain["strike"],
            time_to_expiry=ticker.time_to_expiry,
            risk_free_rate=ticker.risk_free_rate,
            dividend_yield=ticker.dividend_yield,
            volatility=sigma,
            option_type="put"
        )

        scenario_prices["calls"][model_name] = as_aligned_series(
            model_call_prices,
            ticker.call_chain.index,
            f"BS_{model_name}",
        )
        scenario_prices["puts"][model_name] = as_aligned_series(
            model_put_prices,
            ticker.put_chain.index,
            f"BS_{model_name}",
        )

    return scenario_prices




def build_comparison_table(
    chain: pd.DataFrame,
    model_prices: Mapping[str, pd.Series],
    ticker: OptionTicker,
) -> pd.DataFrame:
    """Build one market-versus-model comparison table."""

    base_columns = [
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "lastTradeDate",
        "impliedVolatility",
        "volume",
        "openInterest",
    ]

    comparison = chain.reindex(columns=base_columns).copy()
    comparison["Current Stock Price"] = ticker.current_price
    comparison["Expiry"] = ticker.expiry
    comparison["MarketMid"] = (
        comparison["bid"] + comparison["ask"]
    ) / 2

    for model_name, prices in model_prices.items():
        price_column = f"BS_{model_name}"

        if model_name == "IV":
            comparison["IV Used"] = comparison["impliedVolatility"]
        else:
            comparison[f"{model_name} Used"] = (
                ticker.volatility_inputs[model_name]
            )

        comparison[price_column] = prices.reindex(comparison.index)

        for market_column, market_name in (
            ("MarketMid", "Mid"),
            ("ask", "Ask"),
            ("bid", "Bid"),
        ):
            comparison[f"{price_column} - {market_name}"] = (
                comparison[price_column]
                - comparison[market_column]
            )
            comparison[f"{price_column} {market_name}Edge"] = (
                safe_relative_edge(
                    comparison[price_column],
                    comparison[market_column],
                )
            )

    return comparison



# ---------------------------------------------------------------------------
# Put-call parity
# ---------------------------------------------------------------------------



def check_put_call_parity(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    ticker: OptionTicker,
    tolerance: Optional[float] = None,
) -> pd.DataFrame:
    """Check European put-call parity using the ticker object's inputs."""

    tolerance = (
        ticker.parity_tolerance
        if tolerance is None
        else tolerance
    )

    call_values = (
        calls[["strike", "BS_ForeV"]]
        .dropna()
        .drop_duplicates(subset="strike")
        .rename(columns={"BS_ForeV": "Call_BS_ForeV"})
    )
    put_values = (
        puts[["strike", "BS_ForeV"]]
        .dropna()
        .drop_duplicates(subset="strike")
        .rename(columns={"BS_ForeV": "Put_BS_ForeV"})
    )

    parity = call_values.merge(
        put_values,
        on="strike",
        how="inner",
        validate="one_to_one",
    )

    if parity.empty:
        logger.warning(
            "%s has no matching call and put strikes for parity checking",
            ticker.symbol,
        )
        return parity

    parity["Theoretical C-P"] = (
        ticker.current_price
        * np.exp(
            -ticker.dividend_yield
            * ticker.time_to_expiry
        )
        - parity["strike"]
        * np.exp(
            -ticker.risk_free_rate
            * ticker.time_to_expiry
        )
    )
    parity["Observed C-P"] = (
        parity["Call_BS_ForeV"]
        - parity["Put_BS_ForeV"]
    )
    parity["Parity Error"] = (
        parity["Observed C-P"]
        - parity["Theoretical C-P"]
    )
    parity["Absolute Parity Error"] = parity["Parity Error"].abs()
    parity["Within Tolerance"] = (
        parity["Absolute Parity Error"] <= tolerance
    )

    maximum_error = float(parity["Absolute Parity Error"].max())
    failed_count = int((~parity["Within Tolerance"]).sum())

    logger.info(
        "%s put-call parity check: %d matched strikes, "
        "maximum absolute error %.10f",
        ticker.symbol,
        len(parity),
        maximum_error,
    )

    if failed_count:
        logger.warning(
            "%s has %d strikes outside the %.2e parity tolerance",
            ticker.symbol,
            failed_count,
            tolerance,
        )

    ticker.parity_table = parity
    return parity



def attach_parity_errors(
    options: pd.DataFrame,
    parity: pd.DataFrame,
) -> pd.DataFrame:
    """Attach parity-error columns to a call or put comparison table."""

    if parity.empty:
        result = options.copy()
        result["Parity Error"] = np.nan
        result["Absolute Parity Error"] = np.nan
        result["Parity Valid"] = False
        return result

    parity_columns = parity[[
        "strike",
        "Parity Error",
        "Absolute Parity Error",
        "Within Tolerance",
    ]].rename(columns={"Within Tolerance": "Parity Valid"})

    return options.merge(
        parity_columns,
        on="strike",
        how="left",
        validate="many_to_one",
    )


# ---------------------------------------------------------------------------
# Recommendations and market-quality filters
# ---------------------------------------------------------------------------



def add_recommendation_metrics(
    options: pd.DataFrame,
    buy_edge: float,
) -> pd.DataFrame:
    """Add spread, moneyness and eligibility-aware recommendations."""

    result = options.copy()

    valid_midpoint = result["MarketMid"] > 0
    result["SpreadPct"] = np.nan
    result.loc[valid_midpoint, "SpreadPct"] = (
        result.loc[valid_midpoint, "ask"]
        - result.loc[valid_midpoint, "bid"]
    ) / result.loc[valid_midpoint, "MarketMid"]

    result["Moneyness"] = (
        result["strike"]
        / result["Current Stock Price"]
    )

    edge = result["BS_ForeV AskEdge"]
    missing_quote = result["bid"].isna() | result["ask"].isna()
    quote_valid = result["Quote Valid"].fillna(False)
    eligible = result["Recommendation Eligible"].fillna(False)

    result["Initial Recommended Action"] = np.select(
        condlist=[
            missing_quote,
            ~quote_valid,
            ~eligible,
            edge.isna(),
            edge >= buy_edge,
            (edge > 0) & (edge < buy_edge),
            edge <= 0,
        ],
        choicelist=[
            "No Data",
            "Invalid Quote",
            "Ineligible",
            "No Data",
            "Buy",
            "Positive Edge",
            "Do Not Buy",
        ],
        default="No Data",
    )

    return result




def filter_highlighted_options(
    options: pd.DataFrame,
    buy_edge: float,
    max_spread_pct: float = MAX_SPREAD_PCT,
    min_ask: float = MIN_ASK,
    min_open_interest: int = MIN_OPEN_INTEREST,
    min_volume: int = MIN_VOLUME,
    min_moneyness: float = MIN_MONEYNESS,
    max_moneyness: float = MAX_MONEYNESS,
) -> pd.DataFrame:
    """Return eligible, liquid contracts with the required forecast edge."""

    mask = (
        options["Recommendation Eligible"].eq(True)
        & options["Initial Recommended Action"].eq("Buy")
        & options["bid"].notna()
        & options["ask"].notna()
        & (options["SpreadPct"] <= max_spread_pct)
        & (options["ask"] >= min_ask)
        & (options["openInterest"].fillna(0) >= min_open_interest)
        & (options["volume"].fillna(0) >= min_volume)
        & (options["Moneyness"] >= min_moneyness)
        & (options["Moneyness"] <= max_moneyness)
        & (options["BS_ForeV AskEdge"] >= buy_edge)
    )

    return options.loc[mask].copy()

def option_price_bounds(
    ticker,
    strike,
    option_type,
):
    spot = ticker.current_price
    time_to_expiry = ticker.time_to_expiry
    risk_free_rate = ticker.risk_free_rate
    dividend_yield = ticker.dividend_yield

    discounted_spot = (
        spot
        * np.exp(-dividend_yield * time_to_expiry)
    )

    discounted_strike = (
        strike
        * np.exp(-risk_free_rate * time_to_expiry)
    )

    if option_type == "call":
        lower_bound = max(
            discounted_spot - discounted_strike,
            0.0,
        )

        upper_bound = discounted_spot

    elif option_type == "put":
        lower_bound = max(
            discounted_strike - discounted_spot,
            0.0,
        )

        upper_bound = discounted_strike

    else:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    return lower_bound, upper_bound


def price_option_universe(
    tickers: list[OptionTicker],
    data: pd.DataFrame,
    raise_on_error: bool = True,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """Price and validate all ticker objects using their stored assumptions."""

    cleaned_options: dict[str, pd.DataFrame] = {}
    highlighted_options: dict[str, pd.DataFrame] = {}
    parity_results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    logger.info(
        "Starting option pricing for %d ticker objects",
        len(tickers),
    )

    for ticker in tickers:
        stage = "checking attached option chains"

        if ticker.call_chain is None or ticker.put_chain is None:
            logger.warning(
                "%s skipped because its option chains are unavailable",
                ticker.symbol,
            )
            continue

        logger.info(
            "%s beginning option-chain pricing",
            ticker.symbol,
        )

        try:
            stage = "reading current stock price"
            ticker.set_current_price(
                latest_feature_value(
                    data=data,
                    feature="Close",
                    ticker=ticker,
                )
            )

            stage = "collecting volatility inputs"
            get_volatility_inputs(data=data, ticker=ticker)

            stage = "calculating Black-Scholes scenarios"
            scenario_prices = calculate_black_scholes_scenarios(ticker)

            stage = "building comparison tables"
            calls = build_comparison_table(
                chain=ticker.call_chain,
                model_prices=scenario_prices["calls"],
                ticker=ticker,
            )
            puts = build_comparison_table(
                chain=ticker.put_chain,
                model_prices=scenario_prices["puts"],
                ticker=ticker,
            )

            # Calculate bid, midpoint and ask implied volatility
            stage = "calculating market implied volatilities"

            calls = add_implied_volatility_columns(
                options=calls,
                ticker=ticker,
                option_type="call",
            )

            puts = add_implied_volatility_columns(
                options=puts,
                ticker=ticker,
                option_type="put",
            )

            calls["BS_From_IV_Mid"] = black_scholes(
                ticker=ticker,
                spot=ticker.current_price,
                strike=calls["strike"],
                time_to_expiry=ticker.time_to_expiry,
                risk_free_rate=ticker.risk_free_rate,
                dividend_yield=ticker.dividend_yield,
                option_type="call",
                volatility=calls["IV_Mid"],
)

            puts["BS_From_IV_Mid"] = black_scholes(
                ticker=ticker,
                spot=ticker.current_price,
                strike=puts["strike"],
                time_to_expiry=ticker.time_to_expiry,
                risk_free_rate=ticker.risk_free_rate,
                dividend_yield=ticker.dividend_yield,
                option_type="put",
                volatility=puts["IV_Mid"],
            )


            puts["IV_Repricing_Error"] = puts["BS_From_IV_Mid"] - puts["MarketMid"]
            calls["IV_Repricing_Error"] = calls["BS_From_IV_Mid"] - calls["MarketMid"]

            calls["IV_Repricing_Valid"] = abs(calls["IV_Repricing_Error"]) <= 1e-4
            puts["IV_Repricing_Valid"] = abs(puts["IV_Repricing_Error"]) <= 1e-4

            stage = "calculating analytical Greeks"

            calls = add_greek_columns(
                options=calls,
                ticker=ticker,
                option_type="call",
            )

            puts = add_greek_columns(
                options=puts,
                ticker=ticker,
                option_type="put",
            )

            
            logger.info(
                "%s implied volatility calculations complete",
                ticker.symbol,
            )

            stage = "checking put-call parity"
            parity = check_put_call_parity(
                calls=calls,
                puts=puts,
                ticker=ticker,
            )
            parity_results[ticker.symbol] = parity
            calls = attach_parity_errors(calls, parity)
            puts = attach_parity_errors(puts, parity)

            stage = "adding shared validation information"

            for option_type, options in (
                ("Calls", calls),
                ("Puts", puts),
            ):
                options["Ticker"] = ticker.symbol
                options["Option_Type"] = option_type
                options["Current Stock Price"] = ticker.current_price
                options["Expiry"] = ticker.expiry
                options["Calendar DTE"] = ticker.calendar_dte
                options["Trading DTE"] = ticker.trading_dte
                options["Forecast Horizon"] = ticker.forecast_horizon
                options["Horizon Difference"] = (
                    ticker.trading_dte
                    - ticker.forecast_horizon
                )
                options["Horizon Aligned"] = (
                    options["Horizon Difference"].abs().le(3)
                )
                options["Time to Expiry"] = ticker.time_to_expiry
                options["Risk-Free Rate"] = ticker.risk_free_rate
                options["Dividend Yield"] = ticker.dividend_yield
                options["Forward Price"] = ticker.forward_price
                options["Forward Moneyness"] = (
                    options["strike"] / ticker.forward_price
                )
                options["Volatility Spread"] = (
                    options["ForeV Used"] - options["IV Used"]
                )

                options["Moneyness"] = (
                    options["strike"] / ticker.current_price
                )
                valid_midpoint = options["MarketMid"] > 0
                options["SpreadPct"] = np.nan
                options.loc[valid_midpoint, "SpreadPct"] = (
                    options.loc[valid_midpoint, "ask"]
                    - options.loc[valid_midpoint, "bid"]
                ) / options.loc[valid_midpoint, "MarketMid"]

                discounted_strike = (
                    options["strike"]
                    * np.exp(
                        -ticker.risk_free_rate
                        * ticker.time_to_expiry
                    )
                )

                options["Maximum Buy Price"] = (
                    options["BS_ForeV"]
                    / (1 + ticker.buy_edge)
                )
                options["Ask Below Maximum"] = (
                    options["ask"].notna()
                    & (
                        options["ask"]
                        <= options["Maximum Buy Price"]
                    )
                )

                if option_type == "Calls":
                    options["Intrinsic Value"] = (
                        ticker.current_price - options["strike"]
                    ).clip(lower=0)
                    options["Lower Price Bound"] = (
                        ticker.discounted_spot - discounted_strike
                    ).clip(lower=0)
                    options["Upper Price Bound"] = ticker.discounted_spot
                    options["Break-Even Price"] = (
                        options["strike"] + options["ask"]
                    )
                else:
                    options["Intrinsic Value"] = (
                        options["strike"] - ticker.current_price
                    ).clip(lower=0)
                    options["Lower Price Bound"] = (
                        discounted_strike - ticker.discounted_spot
                    ).clip(lower=0)
                    options["Upper Price Bound"] = discounted_strike
                    options["Break-Even Price"] = (
                        options["strike"] - options["ask"]
                    )

                options["Time Value"] = (
                    options["MarketMid"]
                    - options["Intrinsic Value"]
                )
                options["Distance Above Lower Bound"] = (
                    options["MarketMid"]
                    - options["Lower Price Bound"]
                )
                options["Distance Below Upper Bound"] = (
                    options["Upper Price Bound"]
                    - options["MarketMid"]
                )
                options["Pricing Bounds Valid"] = (
                    options["MarketMid"].notna()
                    & (
                        options["MarketMid"]
                        >= options["Lower Price Bound"] - 1e-6
                    )
                    & (
                        options["MarketMid"]
                        <= options["Upper Price Bound"] + 1e-6
                    )
                )

                if "lastTradeDate" in options.columns:
                    last_trade_date = (
                        pd.to_datetime(
                            options["lastTradeDate"],
                            errors="coerce",
                            utc=True,
                        )
                        .dt.tz_convert(None)
                        .dt.normalize()
                    )
                    options["Quote Age Days"] = (
                        ticker.valuation_date - last_trade_date
                    ).dt.days
                    quote_is_recent = (
                        options["Quote Age Days"]
                        .le(5)
                        .fillna(False)
                    )
                else:
                    options["Quote Age Days"] = np.nan
                    quote_is_recent = pd.Series(
                        True,
                        index=options.index,
                    )

                valid_bid_ask = (
                    options["bid"].notna()
                    & options["ask"].notna()
                    & options["bid"].ge(0)
                    & options["ask"].gt(0)
                    & options["ask"].ge(options["bid"])
                )

                options["Quote Valid"] = (
                    valid_bid_ask
                    & options["Pricing Bounds Valid"]
                )
                options["Quote Issue"] = np.select(
                    [
                        options["bid"].isna() | options["ask"].isna(),
                        options["bid"].lt(0),
                        options["ask"].le(0),
                        options["ask"].lt(options["bid"]),
                        ~options["Pricing Bounds Valid"],
                    ],
                    [
                        "Missing bid or ask",
                        "Negative bid",
                        "Non-positive ask",
                        "Ask below bid",
                        "Outside pricing bounds",
                    ],
                    default="",
                )

                # Put-call parity validates the pricing implementation; it is
                # not a liquidity or quote-quality requirement for one contract.
                options["Recommendation Eligible"] = (
                    options["Quote Valid"]
                    & options["Horizon Aligned"].eq(True)
                    & options["BS_ForeV"].notna()
                    & options["ForeV Used"].notna()
                    & options["ask"].notna()
                )

                options["Greek Volatility"] = (
                    options["IV_Mid"]
                    .fillna(options["IV Used"])
                    .fillna(options["ForeV Used"])
                )

                options["Greek Volatility Source"] = np.select(
                    [
                        options["IV_Mid"].notna(),
                        options["IV Used"].notna(),
                        options["ForeV Used"].notna(),
                    ],
                    [
                        "Calculated Mid IV",
                        "Downloaded IV",
                        "Forecast RV",
                    ],
                    default="No Volatility",
                )

            stage = "adding recommendations and filters"
            calls = add_recommendation_metrics(
                options=calls,
                buy_edge=ticker.buy_edge,
            )
            puts = add_recommendation_metrics(
                options=puts,
                buy_edge=ticker.buy_edge,
            )

            highlighted_calls = filter_highlighted_options(
                options=calls,
                buy_edge=ticker.buy_edge,
            )
            highlighted_puts = filter_highlighted_options(
                options=puts,
                buy_edge=ticker.buy_edge,
            )

            ticker.call_analysis = calls
            ticker.put_analysis = puts

            cleaned_options[ticker.table_key("Calls")] = calls
            cleaned_options[ticker.table_key("Puts")] = puts
            highlighted_options[ticker.table_key("Calls")] = highlighted_calls
            highlighted_options[ticker.table_key("Puts")] = highlighted_puts

            logger.info(
                "%s pricing complete — %d total calls, %d total puts, "
                "%d highlighted calls and %d highlighted puts",
                ticker.symbol,
                len(calls),
                len(puts),
                len(highlighted_calls),
                len(highlighted_puts),
            )

        except Exception as error:
            failures[ticker.symbol] = (
                f"{type(error).__name__}: {error}"
            )
            logger.exception(
                "%s failed during stage '%s'",
                ticker.symbol,
                stage,
            )
            if raise_on_error:
                raise

    logger.info(
        "Option pricing finished — %d full tables, %d highlighted "
        "tables, %d parity tables and %d failed tickers",
        len(cleaned_options),
        len(highlighted_options),
        len(parity_results),
        len(failures),
    )

    if not cleaned_options:
        raise RuntimeError(
            "No option tables were created. Failures: "
            f"{failures}"
        )

    return cleaned_options, highlighted_options, parity_results



def print_highlighted_options(
    highlighted_options: Mapping[str, pd.DataFrame],
) -> None:
    """Print a compact view of the highlighted contracts."""

    important_columns = [
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "BS_ForeV",
        "BS_ForeV AskEdge",
        "Moneyness",
        "MarketMid",
        "SpreadPct",
        "volume",
        "openInterest",
        "Parity Valid",
        "Initial Recommended Action",
    ]

    for table_name, options in highlighted_options.items():
        available_columns = [
            column
            for column in important_columns
            if column in options.columns
        ]

        print(f"\n{table_name}:")
        print(options[available_columns].round(4))


# ---------------------------------------------------------------------------
# Monte Carlo plots and profit distributions
# ---------------------------------------------------------------------------



def plot_monte_carlo_result(
    ticker: OptionTicker,
    paths: np.ndarray,
    terminal_prices: np.ndarray,
    path_count: int = 100,
) -> None:
    """Plot a sample of paths and the terminal-price distribution."""

    median_path = np.median(paths, axis=1)
    shown_paths = min(path_count, paths.shape[1])

    '''plt.figure()
    plt.plot(paths[:, :shown_paths], alpha=0.4)
    plt.plot(
        median_path,
        linewidth=3,
        label="Median simulation",
    )
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price ($)")
    plt.title(f"{ticker.symbol} Monte Carlo Simulation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(terminal_prices, bins=50)
    plt.axvline(
        ticker.current_price,
        linestyle="--",
        label="Current Price",
    )
    plt.xlabel("Price at Expiry ($)")
    plt.ylabel("Frequency")
    plt.title(f"{ticker.symbol} Price Distribution at Expiry")
    plt.legend()
    plt.tight_layout()
    plt.show()'''



def add_option_profit_distribution(
    options: pd.DataFrame,
    terminal_prices: np.ndarray,
    option_type: str,
    risk_free_rate: float,
    time_to_expiry: float,
    include_premium_carry: bool = True,
) -> pd.DataFrame:
    """Calculate vectorised terminal-profit metrics for every contract.

    The output is per share. Multiply by 100 for a standard US equity option
    contract. When include_premium_carry is True, the premium is grown at the
    risk-free rate to put both the premium and payoff on the expiry-date basis.
    """

    result = options.copy()

    premium = result["ask"].fillna(result["BS_ForeV"])

    result["Premium Used"] = premium
    result["Premium Source"] = np.where(
        result["ask"].notna(),
        "Ask",
        "BS_ForeV",
    )

    premium_multiplier = (
        np.exp(risk_free_rate * time_to_expiry)
        if include_premium_carry
        else 1.0
    )

    premium_at_expiry = premium.to_numpy(dtype=float) * premium_multiplier
    strikes = result["strike"].to_numpy(dtype=float)
    terminal_prices = np.asarray(terminal_prices, dtype=float)

    if option_type.lower() == "call":
        payoffs = np.maximum(
            terminal_prices[:, None] - strikes[None, :],
            0.0,
        )
    elif option_type.lower() == "put":
        payoffs = np.maximum(
            strikes[None, :] - terminal_prices[:, None],
            0.0,
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    profits = payoffs - premium_at_expiry[None, :]

    percentiles = np.percentile(
        profits,
        [5, 50, 95],
        axis=0,
    )

    result["Premium at Expiry"] = premium_at_expiry
    result["Lower Profit"] = percentiles[0]
    result["Median Profit"] = percentiles[1]
    result["Upper Profit"] = percentiles[2]
    result["Expected Profit"] = profits.mean(axis=0)
    result["Probability of Profit"] = (profits > 0).mean(axis=0)

    # Returns normalise contracts with very different premiums and strikes.
    valid_premium = premium_at_expiry > 0

    for profit_column, return_column in (
        ("Lower Profit", "Lower Profit Return"),
        ("Median Profit", "Median Profit Return"),
        ("Upper Profit", "Upper Profit Return"),
        ("Expected Profit", "Expected Profit Return"),
    ):
        result[return_column] = np.nan
        result.loc[valid_premium, return_column] = (
            result.loc[valid_premium, profit_column]
            / premium_at_expiry[valid_premium]
        )

    return result



def run_monte_carlo_analysis(
    tickers: list[OptionTicker],
    cleaned_options: dict[str, pd.DataFrame],
    simulations: int = MONTE_CARLO_SIMULATIONS,
    plot_results: bool = PLOT_MONTE_CARLO,
    random_seed: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    """Run Monte Carlo analysis using each ticker object's stored state."""

    monte_carlo_results: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        call_key = ticker.table_key("Calls")
        put_key = ticker.table_key("Puts")

        if (
            ticker.call_chain is None
            or ticker.put_chain is None
            or call_key not in cleaned_options
            or put_key not in cleaned_options
        ):
            logger.warning(
                "%s skipped in Monte Carlo because required data is unavailable",
                ticker.symbol,
            )
            continue

        logger.info(
            "Running %d Monte Carlo simulations for %s",
            simulations,
            ticker.symbol,
        )

        try:
            result = monte_carlo_option_chain(
                ticker=ticker,
                simulations=simulations,
                random_seed=random_seed,
            )
            paths = np.asarray(result["paths"], dtype=float)
            terminal_prices = np.asarray(
                result["terminal_prices"],
                dtype=float,
            )

            if terminal_prices.ndim != 1 or terminal_prices.size == 0:
                raise ValueError(
                    f"Invalid terminal-price array for {ticker.symbol}: "
                    f"shape {terminal_prices.shape}"
                )

            if plot_results:
                plot_monte_carlo_result(
                    ticker=ticker,
                    paths=paths,
                    terminal_prices=terminal_prices,
                )

            for option_key, result_key, option_type in (
                (call_key, "calls", "call"),
                (put_key, "puts", "put"),
            ):
                mc_columns = result[result_key][[
                    "contractSymbol",
                    "MC_ForeV",
                    "MC_SE",
                    "MC AskEdge",
                ]]

                options = cleaned_options[option_key].merge(
                    mc_columns,
                    on="contractSymbol",
                    how="left",
                    validate="one_to_one",
                )
                options["MC - BS ForeV"] = (
                    options["MC_ForeV"]
                    - options["BS_ForeV"]
                )
                options["MC Difference in SE"] = np.where(
                    options["MC_SE"] > 0,
                    options["MC - BS ForeV"] / options["MC_SE"],
                    np.nan,
                )

                cleaned_options[option_key] = add_option_profit_distribution(
                    options=options,
                    terminal_prices=terminal_prices,
                    option_type=option_type,
                    risk_free_rate=ticker.risk_free_rate,
                    time_to_expiry=ticker.time_to_expiry,
                )

            ticker.call_analysis = cleaned_options[call_key]
            ticker.put_analysis = cleaned_options[put_key]
            ticker.monte_carlo_result = result
            monte_carlo_results[ticker.symbol] = result

            logger.info(
                "%s Monte Carlo profit analysis complete",
                ticker.symbol,
            )

        except Exception:
            logger.exception(
                "Monte Carlo analysis failed for %s",
                ticker.symbol,
            )

    return monte_carlo_results



# ---------------------------------------------------------------------------
# Final combined tables
# ---------------------------------------------------------------------------


def combine_option_tables(
    cleaned_options: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine all ticker/type tables into one DataFrame."""

    if not cleaned_options:
        logger.warning("No cleaned option tables are available to combine")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for table_name, options in cleaned_options.items():
        ticker, option_type = table_name.rsplit(" ", maxsplit=1)

        frames.append(
            options.assign(
                Ticker=ticker,
                Option_Type=option_type,
                Option_Table=table_name,
            )
        )

    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        "Combined %d option tables into %d rows",
        len(frames),
        len(combined),
    )

    return combined


def create_findings_table(
    large_table: pd.DataFrame,
) -> pd.DataFrame:
    """Create a robust grouped summary using medians instead of raw means."""

    required_columns = {
        "Initial Recommended Action",
        "Option_Table",
        "contractSymbol",
        "Lower Profit",
        "Median Profit",
        "Upper Profit",
        "Expected Profit",
        "Lower Profit Return",
        "Median Profit Return",
        "Upper Profit Return",
        "Expected Profit Return",
        "Probability of Profit",
        "BS_ForeV",
        "BS_ForeV AskEdge",
    }

    missing_columns = required_columns.difference(large_table.columns)

    if missing_columns:
        raise KeyError(
            "Cannot create findings table; missing columns: "
            f"{sorted(missing_columns)}"
        )

    findings = (
        large_table
        .groupby(
            ["Initial Recommended Action", "Option_Table"],
            observed=True,
        )
        .agg(
            Option_Count=("contractSymbol", "size"),
            Median_Lower_Profit=("Lower Profit", "median"),
            Median_Central_Profit=("Median Profit", "median"),
            Median_Upper_Profit=("Upper Profit", "median"),
            Median_Expected_Profit=("Expected Profit", "median"),
            Median_Lower_Return=("Lower Profit Return", "median"),
            Median_Central_Return=("Median Profit Return", "median"),
            Median_Upper_Return=("Upper Profit Return", "median"),
            Median_Expected_Return=("Expected Profit Return", "median"),
            Median_Probability_Profit=("Probability of Profit", "median"),
            Median_BS_ForeV=("BS_ForeV", "median"),
            Median_Ask_Edge=("BS_ForeV AskEdge", "median"),
        )
        .reset_index()
    )

    logger.info(
        "Created findings table with %d groups",
        len(findings),
    )

    return findings




def print_final_tables(
    findings_table: pd.DataFrame,
    large_table: pd.DataFrame,
) -> None:
    """Print the grouped findings and recommended-price output."""

    print("\nGrouped Findings:")
    print(findings_table.round(4))

    important_columns = [
        # Contract details
        "Ticker",
        "Option_Type",
        "contractSymbol",
        "Expiry",
        "strike",

        # Horizon details
        "Calendar DTE",
        "Trading DTE",
        "Forecast Horizon",
        "Horizon Difference",
        "Horizon Aligned",

        # Underlying and moneyness
        "Current Stock Price",
        "Moneyness",
        "Forward Price",
        "Forward Moneyness",

        # Market quote
        "bid",
        "ask",
        "MarketMid",
        "SpreadPct",
        "volume",
        "openInterest",

        # Volatility comparison
        "IV Used",
        "ForeV Used",
        "Volatility Spread",

        # Valuation
        "BS_IV",
        "BS_ForeV",
        "BS_ForeV AskEdge",

        # Contract properties
        "Intrinsic Value",
        "Time Value",
        "Break-Even Price",

        # Validation
        "Pricing Bounds Valid",
        "Quote Valid",
        "Quote Issue",
        "Parity Valid",
        "Recommendation Eligible",

        # Monte Carlo results
        "Lower Profit Return",
        "Median Profit Return",
        "Upper Profit Return",
        "Expected Profit Return",
        "Probability of Profit",

        #Implied volatility
        "IV_Order Valid",
        "IV_Repricing_Valid",
        "Greek Volatility",
        "Greek Volatility Source",
        "BS_From_IV_Mid",

        # Greeks
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",


        # Final decision
        "Initial Recommended Action",
    ]

    available_columns = [
        column
        for column in important_columns
        if column in large_table.columns
    ]

    print("\nRecommended Buy Prices:")
    print(large_table[available_columns].round(4))



def calendar_days_to_trading_days(
    calendar_days: int,
    valuation_date: Optional[pd.Timestamp] = None,
) -> int:
    """Convert calendar days into weekdays from one valuation date."""

    if calendar_days < 0:
        raise ValueError("calendar_days cannot be negative")

    if valuation_date is None:
        valuation_date = pd.Timestamp.today().normalize()
    else:
        valuation_date = pd.Timestamp(valuation_date).normalize()

    target_date = valuation_date + pd.Timedelta(days=calendar_days)
    trading_days = int(
        np.busday_count(
            valuation_date.date(),
            target_date.date(),
        )
    )

    logger.info(
        "Converted %d calendar days from %s into %d approximate trading days",
        calendar_days,
        valuation_date.date(),
        trading_days,
    )

    return trading_days

def objective_function(volatility, 
                ticker,
                market_price,
                strike,
                time_to_expiry,
                risk_free_rate,
                dividend_yield,
                option_type
):
        return (
            black_scholes(
                ticker,
                ticker.current_price,
                strike,
                time_to_expiry,
                risk_free_rate,
                dividend_yield,
                option_type,
                volatility
            )
            - market_price
        )


def black_scholes_greeks(
    ticker,
    strike,
    volatility,
    option_type,
    time_to_expiry=None,
):
    """
    Calculate Black-Scholes Greeks for one European option.

    Units
    -----
    Delta:
        Option-price change per $1 stock-price change.

    Gamma:
        Delta change per $1 stock-price change.

    Vega:
        Option-price change per one volatility percentage point.

    Theta:
        Option-price change per one calendar day passing.

    Rho:
        Option-price change per one interest-rate percentage point.
    """

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    spot = float(ticker.current_price)
    strike = float(strike)
    volatility = float(volatility)

    if time_to_expiry is None:
        time_to_expiry = float(ticker.time_to_expiry)
    else:
        time_to_expiry = float(time_to_expiry)

    risk_free_rate = float(ticker.risk_free_rate)
    dividend_yield = float(ticker.dividend_yield)

    if (
        not np.isfinite(spot)
        or spot <= 0
        or not np.isfinite(strike)
        or strike <= 0
        or not np.isfinite(volatility)
        or volatility <= 0
        or not np.isfinite(time_to_expiry)
    ):
        return {
            "Delta": np.nan,
            "Gamma": np.nan,
            "Vega": np.nan,
            "Theta": np.nan,
            "Rho": np.nan,
        }

    # Handle expiry separately
    if time_to_expiry <= 0:

        if option_type == "call":
            if spot > strike:
                delta = 1.0
            elif spot < strike:
                delta = 0.0
            else:
                delta = 0.5

        else:
            if spot < strike:
                delta = -1.0
            elif spot > strike:
                delta = 0.0
            else:
                delta = -0.5

        return {
            "Delta": delta,
            "Gamma": np.nan if spot == strike else 0.0,
            "Vega": 0.0,
            "Theta": 0.0,
            "Rho": 0.0,
        }

    sqrt_time = np.sqrt(time_to_expiry)

    d1 = (
        np.log(spot / strike)
        + (
            risk_free_rate
            - dividend_yield
            + 0.5 * volatility**2
        ) * time_to_expiry
    ) / (
        volatility * sqrt_time
    )

    d2 = (
        d1
        - volatility * sqrt_time
    )

    discounted_spot = np.exp(
        -dividend_yield * time_to_expiry
    )

    discounted_strike = np.exp(
        -risk_free_rate * time_to_expiry
    )

    normal_density = norm.pdf(d1)

    # Same gamma and vega for calls and puts
    gamma = (
        discounted_spot
        * normal_density
        / (
            spot
            * volatility
            * sqrt_time
        )
    )

    raw_vega = (
        spot
        * discounted_spot
        * normal_density
        * sqrt_time
    )

    # Per one volatility percentage point
    vega = raw_vega / 100

    if option_type == "call":

        delta = (
            discounted_spot
            * norm.cdf(d1)
        )

        annual_theta = (
            -(
                spot
                * discounted_spot
                * normal_density
                * volatility
            )
            / (
                2 * sqrt_time
            )
            - risk_free_rate
            * strike
            * discounted_strike
            * norm.cdf(d2)
            + dividend_yield
            * spot
            * discounted_spot
            * norm.cdf(d1)
        )

        raw_rho = (
            strike
            * time_to_expiry
            * discounted_strike
            * norm.cdf(d2)
        )

    else:

        delta = (
            discounted_spot
            * (
                norm.cdf(d1) - 1
            )
        )

        annual_theta = (
            -(
                spot
                * discounted_spot
                * normal_density
                * volatility
            )
            / (
                2 * sqrt_time
            )
            + risk_free_rate
            * strike
            * discounted_strike
            * norm.cdf(-d2)
            - dividend_yield
            * spot
            * discounted_spot
            * norm.cdf(-d1)
        )

        raw_rho = (
            -strike
            * time_to_expiry
            * discounted_strike
            * norm.cdf(-d2)
        )

    # Per calendar day
    theta = annual_theta / 365

    # Per one interest-rate percentage point
    rho = raw_rho / 100

    return {
        "Delta": float(delta),
        "Gamma": float(gamma),
        "Vega": float(vega),
        "Theta": float(theta),
        "Rho": float(rho),
    }


def add_greek_columns(
    options,
    ticker,
    option_type,
):
    """Add analytical Black-Scholes Greeks to an option table."""

    result = options.copy()

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    # Prefer your calculated midpoint IV.
    # Use downloaded IV if midpoint IV is unavailable.
    # Use forecast RV only as a final fallback.
    result["Greek Volatility"] = (
        result["IV_Mid"]
        .fillna(result["IV Used"])
        .fillna(result["ForeV Used"])
    )

    result["Greek Volatility Source"] = np.select(
        [
            result["IV_Mid"].notna(),
            result["IV Used"].notna(),
            result["ForeV Used"].notna(),
        ],
        [
            "Calculated Mid IV",
            "Downloaded IV",
            "Forecast RV",
        ],
        default="No Volatility",
    )

    greek_results = []

    for strike, volatility in zip(
        result["strike"],
        result["Greek Volatility"],
    ):

        if (
            pd.isna(strike)
            or pd.isna(volatility)
        ):
            greeks = {
                "Delta": np.nan,
                "Gamma": np.nan,
                "Vega": np.nan,
                "Theta": np.nan,
                "Rho": np.nan,
            }

        else:
            greeks = black_scholes_greeks(
                ticker=ticker,
                strike=float(strike),
                volatility=float(volatility),
                option_type=option_type,
            )

        greek_results.append(greeks)

    greek_table = pd.DataFrame(
        greek_results,
        index=result.index,
    )

    for greek in [
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]:
        result[greek] = greek_table[greek]

    result = add_greek_sign_checks(
        options=result,
        option_type=option_type,
    )

    return result


def add_greek_sign_checks(
    options,
    option_type,
    tolerance=1e-10,
):
    """Check reliable theoretical Greek signs and ranges."""

    result = options.copy()

    option_type = option_type.lower()

    greek_columns = [
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]

    complete_greeks = (
        result[greek_columns]
        .notna()
        .all(axis=1)
    )

    result["Greek Signs Valid"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    if option_type == "call":

        valid_signs = (
            result["Delta"].between(
                -tolerance,
                1 + tolerance,
            )
            & result["Gamma"].ge(-tolerance)
            & result["Vega"].ge(-tolerance)
            & result["Rho"].ge(-tolerance)
        )

    elif option_type == "put":

        valid_signs = (
            result["Delta"].between(
                -1 - tolerance,
                tolerance,
            )
            & result["Gamma"].ge(-tolerance)
            & result["Vega"].ge(-tolerance)
            & result["Rho"].le(tolerance)
        )

    else:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    result.loc[
        complete_greeks,
        "Greek Signs Valid"
    ] = valid_signs.loc[complete_greeks]

    result["Greek Sign Issue"] = ""

    if option_type == "call":

        result.loc[
            complete_greeks
            & ~result["Delta"].between(
                -tolerance,
                1 + tolerance,
            ),
            "Greek Sign Issue",
        ] = "Call delta outside 0 to 1"

        result.loc[
            complete_greeks
            & result["Rho"].lt(-tolerance),
            "Greek Sign Issue",
        ] = "Call rho is negative"

    else:

        result.loc[
            complete_greeks
            & ~result["Delta"].between(
                -1 - tolerance,
                tolerance,
            ),
            "Greek Sign Issue",
        ] = "Put delta outside -1 to 0"

        result.loc[
            complete_greeks
            & result["Rho"].gt(tolerance),
            "Greek Sign Issue",
        ] = "Put rho is positive"

    result.loc[
        complete_greeks
        & result["Gamma"].lt(-tolerance),
        "Greek Sign Issue",
    ] = "Gamma is negative"

    result.loc[
        complete_greeks
        & result["Vega"].lt(-tolerance),
        "Greek Sign Issue",
    ] = "Vega is negative"

    result.loc[
        ~complete_greeks,
        "Greek Sign Issue",
    ] = "Missing Greek input"

    return result


def scalar_black_scholes_price(
    ticker,
    spot,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield,
    option_type,
    volatility,
):
    """Return one Black-Scholes price as a scalar float."""

    value = black_scholes(
        ticker=ticker,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
        volatility=volatility,
    )

    array = np.asarray(
        value,
        dtype=float,
    )

    if array.size != 1:
        raise ValueError(
            "Numerical Greek validation requires a scalar option price"
        )

    return float(
        array.reshape(-1)[0]
    )

def numerical_black_scholes_greeks(
    ticker,
    strike,
    volatility,
    option_type,
    spot_bump=None,
    volatility_bump=1e-4,
    rate_bump=1e-5,
    time_bump=1 / 365,
):
    """Calculate numerical Greeks using finite differences."""

    option_type = option_type.lower()

    spot = float(ticker.current_price)
    strike = float(strike)
    volatility = float(volatility)

    time_to_expiry = float(
        ticker.time_to_expiry
    )

    risk_free_rate = float(
        ticker.risk_free_rate
    )

    dividend_yield = float(
        ticker.dividend_yield
    )

    if (
        spot <= 0
        or strike <= 0
        or volatility <= 0
        or time_to_expiry <= 0
    ):
        return {
            "Numerical Delta": np.nan,
            "Numerical Gamma": np.nan,
            "Numerical Vega": np.nan,
            "Numerical Theta": np.nan,
            "Numerical Rho": np.nan,
        }

    if spot_bump is None:
        spot_bump = max(
            spot * 1e-4,
            1e-4,
        )

    # Prevent negative spot
    spot_bump = min(
        spot_bump,
        spot * 0.5,
    )

    # Prevent negative volatility
    volatility_bump = min(
        volatility_bump,
        volatility * 0.5,
    )

    # Keep the shorter time positive
    time_bump = min(
        time_bump,
        time_to_expiry * 0.5,
    )

    def price(
        adjusted_spot=spot,
        adjusted_volatility=volatility,
        adjusted_rate=risk_free_rate,
        adjusted_time=time_to_expiry,
    ):
        return scalar_black_scholes_price(
            ticker=ticker,
            spot=adjusted_spot,
            strike=strike,
            time_to_expiry=adjusted_time,
            risk_free_rate=adjusted_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
            volatility=adjusted_volatility,
        )

    base_price = price()

    price_spot_up = price(
        adjusted_spot=spot + spot_bump,
    )

    price_spot_down = price(
        adjusted_spot=spot - spot_bump,
    )

    numerical_delta = (
        price_spot_up
        - price_spot_down
    ) / (
        2 * spot_bump
    )

    numerical_gamma = (
        price_spot_up
        - 2 * base_price
        + price_spot_down
    ) / (
        spot_bump**2
    )

    price_vol_up = price(
        adjusted_volatility=(
            volatility + volatility_bump
        ),
    )

    price_vol_down = price(
        adjusted_volatility=(
            volatility - volatility_bump
        ),
    )

    # First calculate derivative per 1.00 volatility,
    # then convert to one percentage point
    numerical_vega = (
        (
            price_vol_up
            - price_vol_down
        )
        / (
            2 * volatility_bump
        )
        * 0.01
    )

    price_rate_up = price(
        adjusted_rate=(
            risk_free_rate + rate_bump
        ),
    )

    price_rate_down = price(
        adjusted_rate=(
            risk_free_rate - rate_bump
        ),
    )

    # Convert to one percentage-point rate change
    numerical_rho = (
        (
            price_rate_up
            - price_rate_down
        )
        / (
            2 * rate_bump
        )
        * 0.01
    )

    price_less_time = price(
        adjusted_time=(
            time_to_expiry - time_bump
        ),
    )

    price_more_time = price(
        adjusted_time=(
            time_to_expiry + time_bump
        ),
    )

    # Theta is the effect of calendar time passing.
    # This is the negative derivative with respect to T.
    numerical_theta = (
        price_less_time
        - price_more_time
    ) / (
        2
        * time_bump
        * 365
    )

    return {
        "Numerical Delta": float(numerical_delta),
        "Numerical Gamma": float(numerical_gamma),
        "Numerical Vega": float(numerical_vega),
        "Numerical Theta": float(numerical_theta),
        "Numerical Rho": float(numerical_rho),
    }

def add_numerical_greek_validation(
    options,
    ticker,
    option_type,
):
    """Compare analytical Greeks with finite-difference Greeks."""

    result = options.copy()

    numerical_results = []

    for strike, volatility in zip(
        result["strike"],
        result["Greek Volatility"],
    ):

        if (
            pd.isna(strike)
            or pd.isna(volatility)
        ):
            numerical_greeks = {
                "Numerical Delta": np.nan,
                "Numerical Gamma": np.nan,
                "Numerical Vega": np.nan,
                "Numerical Theta": np.nan,
                "Numerical Rho": np.nan,
            }

        else:
            numerical_greeks = (
                numerical_black_scholes_greeks(
                    ticker=ticker,
                    strike=float(strike),
                    volatility=float(volatility),
                    option_type=option_type,
                )
            )

        numerical_results.append(
            numerical_greeks
        )

    numerical_table = pd.DataFrame(
        numerical_results,
        index=result.index,
    )

    for greek in [
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
    ]:
        numerical_column = (
            f"Numerical {greek}"
        )

        result[numerical_column] = (
            numerical_table[
                numerical_column
            ]
        )

        result[f"{greek} Error"] = (
            result[greek]
            - result[numerical_column]
        )

        result[f"{greek} Absolute Error"] = (
            result[f"{greek} Error"]
            .abs()
        )

        settings = (
            GREEK_VALIDATION_TOLERANCES[
                greek
            ]
        )

        valid_comparison = (
            result[greek].notna()
            & result[numerical_column].notna()
        )

        passed = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

        passed.loc[valid_comparison] = np.isclose(
            result.loc[
                valid_comparison,
                greek,
            ],
            result.loc[
                valid_comparison,
                numerical_column,
            ],
            atol=settings["atol"],
            rtol=settings["rtol"],
        )

        result[
            f"{greek} Validation Passed"
        ] = passed

    validation_columns = [
        f"{greek} Validation Passed"
        for greek in [
            "Delta",
            "Gamma",
            "Vega",
            "Theta",
            "Rho",
        ]
    ]

    complete_validation = (
        result[validation_columns]
        .notna()
        .all(axis=1)
    )

    result["All Greeks Valid"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="boolean",
    )

    result.loc[
        complete_validation,
        "All Greeks Valid",
    ] = (
        result.loc[
            complete_validation,
            validation_columns,
        ]
        .all(axis=1)
    )

    return result


def implied_volatility(
    ticker,
    strike,
    market_price,
    option_type,
    lower_volatility=1e-6,
    upper_volatility=5.0,
):
    """Calculate implied volatility using Brent's method."""

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    if (
        not np.isfinite(market_price)
        or market_price <= 0
        or not np.isfinite(strike)
        or strike <= 0
        or ticker.time_to_expiry <= 0
    ):
        return np.nan

    lower_price_bound, upper_price_bound = (
        option_price_bounds(
            ticker=ticker,
            strike=strike,
            option_type=option_type,
        )
    )

    price_tolerance = 1e-8

    if (
        market_price < lower_price_bound - price_tolerance
        or market_price > upper_price_bound + price_tolerance
    ):
        return np.nan

    objective_args = (
        ticker,
        market_price,
        strike,
        ticker.time_to_expiry,
        ticker.risk_free_rate,
        ticker.dividend_yield,
        option_type,
    )

    lower_objective = objective_function(
        lower_volatility,
        *objective_args,
    )

    upper_objective = objective_function(
        upper_volatility,
        *objective_args,
    )

    if (
        not np.isfinite(lower_objective)
        or not np.isfinite(upper_objective)
        or lower_objective * upper_objective > 0
    ):
        return np.nan

    try:
        return float(
            brentq(
                objective_function,
                lower_volatility,
                upper_volatility,
                args=objective_args,
                xtol=1e-6,
                rtol=1e-6,
                maxiter=1000,
            )
        )

    except (ValueError, RuntimeError):
        return np.nan


def add_implied_volatility_columns(
    options: pd.DataFrame,
    ticker,
    option_type: str,
) -> pd.DataFrame:
    """
    Calculate implied volatility from the bid, midpoint and ask
    for every option in one call or put table.
    """

    result = options.copy()

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'"
        )

    market_price_columns = {
        "bid": "IV_Bid",
        "MarketMid": "IV_Mid",
        "ask": "IV_Ask",
    }

    for market_column, iv_column in market_price_columns.items():

        result[iv_column] = [
            implied_volatility(
                ticker=ticker,
                strike=float(strike),
                market_price=float(market_price)
                if pd.notna(market_price)
                else np.nan,
                option_type=option_type,
            )
            for strike, market_price in zip(
                result["strike"],
                result[market_column],
            )
        ]

    # Width of the market's implied-volatility spread
    result["IV Bid-Ask Spread"] = (
        result["IV_Ask"]
        - result["IV_Bid"]
    )

    # Compare your calculated midpoint IV with Yahoo's IV
    result["IV Mid - Yahoo IV"] = (
        result["IV_Mid"]
        - result["IV Used"]
    )

    # Check the expected IV ordering
    result["IV Order Valid"] = (
        result["IV_Bid"].notna()
        & result["IV_Mid"].notna()
        & result["IV_Ask"].notna()
        & result["IV_Bid"].le(result["IV_Mid"] + 1e-8)
        & result["IV_Mid"].le(result["IV_Ask"] + 1e-8)
    )

    return result


def risk_engine_data_prep(large_table: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the large table for risk-engine input.
    This includes renaming columns and selecting relevant fields.
    """

    logger.info(
        "Preparing option universe for the portfolio risk engine: %d rows",
        len(large_table),
    )

    # A fixed seed keeps the hypothetical test portfolio reproducible.
    rng = np.random.default_rng(seed=42)

    # Retain only the market, timing, volatility and Greek fields required by
    # the position and scenario calculations.
    portfolio_market_columns = [
        # Contract identity
        "contractSymbol",
        "Ticker",
        "Option_Type",
        "strike",
        "Expiry",

        # Current underlying and timing
        "Current Stock Price",
        "Time to Expiry",
        "Calendar DTE",

        # Current option market
        "bid",
        "ask",
        "MarketMid",

        # Volatility used for risk
        "Greek Volatility",
        "Greek Volatility Source",

        # Unit Greeks
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",

        # Quote reliability
        "SpreadPct",
        "volume",
        "openInterest",
        "Quote Valid",
        "Quote Issue",
    ]

    position_options = large_table[
        portfolio_market_columns
    ].copy()

    rows_before_drop = len(position_options)

    # The temporary test portfolio uses only rows with all required inputs.
    # This prevents missing prices, volatilities or Greeks from entering the
    # position calculations.
    position_options = (
        position_options
        .dropna()
        .reset_index(drop=True)
    )

    number_of_positions = len(position_options)

    logger.info(
        "Risk-engine input cleaned: %d usable option rows; %d rows removed",
        number_of_positions,
        rows_before_drop - number_of_positions,
    )

    # Randomly assign long or short
    position_options["Side"] = rng.choice(
        ["long", "short"],
        size=number_of_positions,
    )

    # Random number of contracts from 1 to 5
    position_options["Quantity"] = rng.integers(
        low=1,
        high=6,
        size=number_of_positions,
    )

    # Standard US equity-option contract multiplier
    position_options["Multiplier"] = 100

    # Hypothetical entry prices within 15% of today's midpoint
    entry_price_change = rng.uniform(
        low=-0.15,
        high=0.15,
        size=number_of_positions,
    )

    position_options["Entry Price"] = (
        position_options["MarketMid"]
        * (1 + entry_price_change)
    ).clip(lower=0.01)

    position_options["Current Mark"] = (
        position_options["MarketMid"]
    )

    position_options["Direction"] = np.where(
        position_options["Side"] == "long",
        1,
        -1,
    )

    logger.info(
        "Created reproducible test option positions: %d long and %d short",
        int(position_options["Side"].eq("long").sum()),
        int(position_options["Side"].eq("short").sum()),
    )

    # Add one ordinary-share position per ticker so the engine can represent
    # stock-and-option portfolios and include share delta in total exposure.
    share_holdings = {
        "AAPL": 50,
        "MSFT": -20,
        "META": 100,
        "WMT": 40,
        "GOOGL": 35,
        "AMZN": -40,
        "HCA": 25,
    }

    spot_prices = (
        position_options
        .groupby("Ticker")["Current Stock Price"]
        .first()
    )

    share_rows = pd.DataFrame(
        [
            {
                "contractSymbol": f"{ticker}_SHARES",
                "Ticker": ticker,
                "Option_Type": "Shares",
                "Current Stock Price": spot_prices.loc[ticker],
                "Side": "long" if shares > 0 else "short",
                "Direction": 1 if shares > 0 else -1,
                "Quantity": abs(shares),
                "Multiplier": 1,
                "Entry Price": spot_prices.loc[ticker],
                "Current Mark": spot_prices.loc[ticker],

                # Greeks for ordinary shares
                "Delta": 1.0,
                "Gamma": 0.0,
                "Vega": 0.0,
                "Theta": 0.0,
                "Rho": 0.0,
            }
            for ticker, shares in share_holdings.items()
            if shares != 0
        ]
    )

    # Add all missing option-specific columns and match column order
    share_rows = share_rows.reindex(
        columns=position_options.columns
    )

    # Add share positions underneath the option positions.
    position_options = pd.concat(
        [
            position_options,
            share_rows,
        ],
        ignore_index=True,
    )

    logger.info(
        "Added %d share positions; combined portfolio now contains %d rows",
        len(share_rows),
        len(position_options),
    )

    # Position Scale is signed. It includes direction, number of contracts or
    # shares, and the contract multiplier.
    position_options["Position Scale"] = (
        position_options["Direction"]
        * position_options["Quantity"]
        * position_options["Multiplier"]
    )

    # Convert unit Greeks into signed position-level exposures.
    position_options["Position Delta"] = (
        position_options["Delta"] * position_options["Position Scale"]
    )
    position_options["Position Gamma"] = (
        position_options["Gamma"] * position_options["Position Scale"]
    )
    position_options["Position Vega"] = (
        position_options["Vega"] * position_options["Position Scale"]
    )
    position_options["Position Theta"] = (
        position_options["Theta"] * position_options["Position Scale"]
    )
    position_options["Position Rho"] = (
        position_options["Rho"] * position_options["Position Scale"]
    )

    # Mark-to-market P&L since the hypothetical entry price.
    position_options["Position PnL"] = (
        position_options["Current Mark"]
        - position_options["Entry Price"]
    ) * position_options["Position Scale"]

    position_options["Signed Market Value"] = (
        position_options["Current Mark"]
        * position_options["Position Scale"]
    )

    position_options["Entry Cash Flow"] = (
        position_options["Entry Price"]
        * position_options["Position Scale"]
    )

    position_options["Entry Premium Value"] = (
        position_options["Entry Price"]
        * position_options["Quantity"]
        * position_options["Multiplier"]
    )

    logger.info(
        "Portfolio positions prepared: gross market value %.2f; current P&L %.2f",
        float(position_options["Signed Market Value"].abs().sum()),
        float(position_options["Position PnL"].sum()),
    )

    return position_options

def risk_engine_summary(position_options: pd.DataFrame) -> pd.DataFrame:
    """Aggregate current position value, P&L and Greeks by ticker."""

    logger.info(
        "Building current portfolio risk summary from %d positions",
        len(position_options),
    )

    # First create one current-risk row for every underlying ticker.
    ticker_risk = (
        position_options
        .groupby(
            "Ticker",
            as_index=False,
        )
        .agg(
            Number_of_Positions=(
                "contractSymbol",
                "count",
            ),
            Net_Market_Value=(
                "Signed Market Value",
                "sum",
            ),
            Gross_Market_Value=(
                "Signed Market Value",
                lambda values: values.abs().sum(),
            ),
            Position_PnL=(
                "Position PnL",
                "sum",
            ),
            Delta=(
                "Position Delta",
                "sum",
            ),
            Gamma=(
                "Position Gamma",
                "sum",
            ),
            Vega=(
                "Position Vega",
                "sum",
            ),
            Theta=(
                "Position Theta",
                "sum",
            ),
            Rho=(
                "Position Rho",
                "sum",
            ),
        )
    )

    logger.info(
        "Ticker-level risk calculated for %d tickers",
        len(ticker_risk),
    )

    # Add a final row containing the whole portfolio's current exposure.
    portfolio_total = pd.DataFrame(
        [
            {
                "Ticker": "PORTFOLIO",
                "Number_of_Positions": (
                    ticker_risk["Number_of_Positions"]
                    .sum()
                ),
                "Net_Market_Value": (
                    ticker_risk["Net_Market_Value"]
                    .sum()
                ),
                "Gross_Market_Value": (
                    ticker_risk["Gross_Market_Value"]
                    .sum()
                ),
                "Position_PnL": (
                    ticker_risk["Position_PnL"]
                    .sum()
                ),
                "Delta": ticker_risk["Delta"].sum(),
                "Gamma": ticker_risk["Gamma"].sum(),
                "Vega": ticker_risk["Vega"].sum(),
                "Theta": ticker_risk["Theta"].sum(),
                "Rho": ticker_risk["Rho"].sum(),
            }
        ]
    )

    portfolio_risk = pd.concat(
        [
            ticker_risk,
            portfolio_total,
        ],
        ignore_index=True,
    )

    total = portfolio_total.iloc[0]
    logger.info(
        "Portfolio risk summary complete: net value %.2f; gross value %.2f; "
        "delta %.2f",
        float(total["Net_Market_Value"]),
        float(total["Gross_Market_Value"]),
        float(total["Delta"]),
    )

    return portfolio_risk


def generate_random_scenarios(
    number_of_scenarios=1_000,
    max_days_forward=30,
    seed=42,
):
    """Create reproducible random market shocks plus an unchanged base case."""

    logger.info(
        "Generating %d random scenarios with up to %d days forward; seed=%s",
        number_of_scenarios,
        max_days_forward,
        seed,
    )

    rng = np.random.default_rng(seed)

    scenarios = pd.DataFrame(
        {
            # Percentage change in the underlying price
            "Spot Shock": np.clip(
                rng.normal(
                    loc=0.0,
                    scale=0.08,
                    size=number_of_scenarios,
                ),
                -0.30,
                0.30,
            ),

            # Absolute volatility change:
            # 0.05 means volatility increases by 5 points
            "Volatility Shock": np.clip(
                rng.normal(
                    loc=0.0,
                    scale=0.05,
                    size=number_of_scenarios,
                ),
                -0.20,
                0.20,
            ),

            # Random number of calendar days passing
            "Days Forward": rng.integers(
                low=0,
                high=max_days_forward + 1,
                size=number_of_scenarios,
            ),

            # Absolute interest-rate change:
            # 0.01 means rates increase by 1 percentage point
            "Rate Shock": np.clip(
                rng.normal(
                    loc=0.0,
                    scale=0.005,
                    size=number_of_scenarios,
                ),
                -0.02,
                0.02,
            ),
        }
    )

    scenarios["Scenario ID"] = (
        "R"
        + (
            scenarios.index + 1
        )
        .astype(str)
        .str.zfill(4)
    )

    # Include an unchanged scenario for validation
    base_scenario = pd.DataFrame(
        [
            {
                "Spot Shock": 0.0,
                "Volatility Shock": 0.0,
                "Days Forward": 0,
                "Rate Shock": 0.0,
                "Scenario ID": "BASE",
            }
        ]
    )

    scenario_table = pd.concat(
        [
            base_scenario,
            scenarios,
        ],
        ignore_index=True,
    )

    logger.info(
        "Scenario generation complete: %d total rows including BASE",
        len(scenario_table),
    )
    logger.debug(
        "Scenario ranges — spot: [%.4f, %.4f], volatility: [%.4f, %.4f], "
        "rate: [%.4f, %.4f], days: [%d, %d]",
        float(scenario_table["Spot Shock"].min()),
        float(scenario_table["Spot Shock"].max()),
        float(scenario_table["Volatility Shock"].min()),
        float(scenario_table["Volatility Shock"].max()),
        float(scenario_table["Rate Shock"].min()),
        float(scenario_table["Rate Shock"].max()),
        int(scenario_table["Days Forward"].min()),
        int(scenario_table["Days Forward"].max()),
    )

    return scenario_table


def run_scenario_engine(
    position_options: pd.DataFrame,
    scenarios: pd.DataFrame,
    dividend_yields: dict[str, float],
):
    """Fully revalue every portfolio position under every market scenario."""

    logger.info(
        "Starting scenario engine: %d positions across %d scenarios",
        len(position_options),
        len(scenarios),
    )

    scenario_frames = []

    # Convert Calls, Puts and Shares to call, put and share
    position_types = (
        position_options["Option_Type"]
        .astype(str)
        .str.lower()
        .str.rstrip("s")
    )

    call_mask = position_types.eq("call")
    put_mask = position_types.eq("put")
    share_mask = position_types.eq("share")

    unknown_type_mask = ~(call_mask | put_mask | share_mask)
    if unknown_type_mask.any():
        unknown_types = (
            position_options.loc[unknown_type_mask, "Option_Type"]
            .drop_duplicates()
            .tolist()
        )
        logger.warning(
            "Scenario engine found unrecognised position types: %s",
            unknown_types,
        )

    logger.info(
        "Scenario positions classified: %d calls, %d puts and %d shares",
        int(call_mask.sum()),
        int(put_mask.sum()),
        int(share_mask.sum()),
    )

    # Map each ticker to the dividend yield stored on its ticker object.
    raw_dividend_yield = position_options["Ticker"].map(dividend_yields)

    missing_yield_tickers = (
        position_options.loc[raw_dividend_yield.isna(), "Ticker"]
        .drop_duplicates()
        .tolist()
    )
    if missing_yield_tickers:
        logger.warning(
            "Missing dividend yields defaulted to zero for: %s",
            missing_yield_tickers,
        )

    position_dividend_yield = raw_dividend_yield.fillna(0.0)

    total_scenarios = len(scenarios)
    progress_interval = max(total_scenarios // 10, 1)


    # Signed number of option units or shares
    position_scale = (
        position_options["Quantity"]
        * position_options["Direction"]
        * position_options["Multiplier"]
    )

    # Calculate the model value before applying any scenario shocks
    base_model_price = pd.Series(
        np.nan,
        index=position_options.index,
        dtype=float,
    )

    # A share's base model price is its current stock price
    base_model_price.loc[share_mask] = (
        position_options.loc[
            share_mask,
            "Current Stock Price",
        ]
    )

    # Base Black-Scholes value for calls
    if call_mask.any():
        base_model_price.loc[call_mask] = np.asarray(
            black_scholes(
                ticker=position_options.loc[
                    call_mask,
                    "Ticker",
                ],
                spot=position_options.loc[
                    call_mask,
                    "Current Stock Price",
                ],
                strike=position_options.loc[
                    call_mask,
                    "strike",
                ],
                time_to_expiry=position_options.loc[
                    call_mask,
                    "Time to Expiry",
                ],
                risk_free_rate=position_options.loc[
                    call_mask,
                    "Risk Free Rate",
                ],
                dividend_yield=position_dividend_yield.loc[
                    call_mask
                ],
                option_type="call",
                volatility=position_options.loc[
                    call_mask,
                    "Greek Volatility",
                ],
            )
        )

    # Base Black-Scholes value for puts
    if put_mask.any():
        base_model_price.loc[put_mask] = np.asarray(
            black_scholes(
                ticker=position_options.loc[
                    put_mask,
                    "Ticker",
                ],
                spot=position_options.loc[
                    put_mask,
                    "Current Stock Price",
                ],
                strike=position_options.loc[
                    put_mask,
                    "strike",
                ],
                time_to_expiry=position_options.loc[
                    put_mask,
                    "Time to Expiry",
                ],
                risk_free_rate=position_options.loc[
                    put_mask,
                    "Risk Free Rate",
                ],
                dividend_yield=position_dividend_yield.loc[
                    put_mask
                ],
                option_type="put",
                volatility=position_options.loc[
                    put_mask,
                    "Greek Volatility",
                ],
            )
        )

    logger.info(
        "Base model prices calculated for %d positions",
        base_model_price.notna().sum(),
    )

    for scenario_number, (_, scenario) in enumerate(
        scenarios.iterrows(),
        start=1,
    ):
        if (
            scenario_number == 1
            or scenario_number % progress_interval == 0
            or scenario_number == total_scenarios
        ):
            logger.info(
                "Revaluing scenario %d/%d: %s",
                scenario_number,
                total_scenarios,
                scenario["Scenario ID"],
            )
        # Percentage shock to the underlying
        shocked_spot = (
            position_options["Current Stock Price"]
            * (1 + scenario["Spot Shock"])
        )

        # Absolute volatility shock
        shocked_volatility = (
            position_options["Greek Volatility"]
            + scenario["Volatility Shock"]
        ).clip(lower=1e-8)

        # Time to expiry is already measured in years
        shocked_time = (
            position_options["Time to Expiry"]
            - scenario["Days Forward"] / 365
        ).clip(lower=0)

        # Absolute interest-rate shock
        shocked_rate = (
            position_options["Risk Free Rate"]
            + scenario["Rate Shock"]
        )

        # One scenario price for every portfolio row
        scenario_price = pd.Series(
            np.nan,
            index=position_options.index,
            dtype=float,
        )

        # Shares are worth the shocked stock price
        scenario_price.loc[share_mask] = (
            shocked_spot.loc[share_mask]
        )

        # Price calls
        if call_mask.any():
            scenario_price.loc[call_mask] = np.asarray(
                black_scholes(
                    ticker=position_options.loc[
                        call_mask,
                        "Ticker",
                    ],
                    spot=shocked_spot.loc[call_mask],
                    strike=position_options.loc[
                        call_mask,
                        "strike",
                    ],
                    time_to_expiry=shocked_time.loc[
                        call_mask
                    ],
                    risk_free_rate=shocked_rate.loc[
                        call_mask
                    ],
                    dividend_yield=position_dividend_yield.loc[
                        call_mask
                    ],
                    option_type="call",
                    volatility=shocked_volatility.loc[
                        call_mask
                    ],
                )
            )

        # Price puts
        if put_mask.any():
            scenario_price.loc[put_mask] = np.asarray(
                black_scholes(
                    ticker=position_options.loc[
                        put_mask,
                        "Ticker",
                    ],
                    spot=shocked_spot.loc[put_mask],
                    strike=position_options.loc[
                        put_mask,
                        "strike",
                    ],
                    time_to_expiry=shocked_time.loc[
                        put_mask
                    ],
                    risk_free_rate=shocked_rate.loc[
                        put_mask
                    ],
                    dividend_yield=position_dividend_yield.loc[
                        put_mask
                    ],
                    option_type="put",
                    volatility=shocked_volatility.loc[
                        put_mask
                    ],
                )
            )

        position_scale = (
            position_options["Quantity"]
            * position_options["Direction"]
            * position_options["Multiplier"]
        )

        scenario_frame = pd.DataFrame(
            {
                "Scenario ID": scenario["Scenario ID"],
                "Contract Symbol": (
                    position_options["contractSymbol"]
                ),
                "Ticker": position_options["Ticker"],
                "Option Type": position_options["Option_Type"],
                # Keep the market price and model price separate
                "Current Market Price": (
                    position_options["Current Mark"]
                ),
                "Base Model Price": base_model_price,
                "Scenario Price": scenario_price,

                # Difference between the current market and the model
                "Market-to-Model PnL": (
                    base_model_price
                    - position_options["Current Mark"]
                ) * position_scale,

                # Pure scenario movement measured on a consistent model basis
                "Scenario PnL": (
                    scenario_price
                    - base_model_price
                ) * position_scale,
                "Scenario Market Value": (
                    scenario_price
                    * position_scale
                ),
                "Shocked Spot": shocked_spot,
                "Shocked Volatility": shocked_volatility,
                "Shocked Rate": shocked_rate,
                "Shocked Time": shocked_time,
                "Spot Shock": scenario["Spot Shock"],
                "Volatility Shock": (
                    scenario["Volatility Shock"]
                ),
                "Rate Shock": scenario["Rate Shock"],
                "Days Forward": scenario["Days Forward"],
            }
        )

        missing_prices = int(scenario_price.isna().sum())
        if missing_prices:
            logger.warning(
                "Scenario %s produced %d missing position prices",
                scenario["Scenario ID"],
                missing_prices,
            )

        scenario_frames.append(scenario_frame)

    # Concatenate only once after all scenarios have been processed. This is
    # substantially faster than repeatedly growing one DataFrame in the loop.
    results = pd.concat(
        scenario_frames,
        ignore_index=True,
    )

    logger.info(
        "Position-level scenario results created: %d rows",
        len(results),
    )

    # Summarise each scenario by ticker.
    scenario_ticker = (
        results
        .groupby(
            ["Scenario ID", "Ticker"],
            as_index=False,
        )
        .agg(
            Scenario_PnL=("Scenario PnL", "sum"),
            Scenario_Value=(
                "Scenario Market Value",
                "sum",
            ),
        )
    )

    # Summarise each scenario across the entire portfolio.
    scenario_portfolio = (
        results
        .groupby(
            "Scenario ID",
            as_index=False,
        )
        .agg(
            Portfolio_PnL=("Scenario PnL", "sum"),
            Portfolio_Value=(
                "Scenario Market Value",
                "sum",
            ),
        )
    )

    return (
        results,
        scenario_ticker,
        scenario_portfolio,
    )
        

def analyse_scenario_results(
    scenarios: pd.DataFrame,
    results: pd.DataFrame,
    scenario_ticker: pd.DataFrame,
    scenario_portfolio: pd.DataFrame,
    portfolio_risk: pd.DataFrame,
    base_scenario_id: str = "BASE",
    max_portfolio_loss: float = 100_000,
    max_ticker_loss: float = 25_000,
    number_to_show: int = 10,
):
    """
    Validate and analyse position, ticker and portfolio scenario results.
    """

    logger.info(
        "Analysing scenario outputs: %d position rows, %d ticker rows and "
        "%d portfolio rows",
        len(results),
        len(scenario_ticker),
        len(scenario_portfolio),
    )

    # Work on copies so the raw outputs remain available unchanged.
    results = results.copy()
    scenario_ticker = scenario_ticker.copy()
    scenario_portfolio = scenario_portfolio.copy()

    # Allow the older position-level P&L column name
    if (
        "Scenario PnL" not in results.columns
        and "Profit/Loss" in results.columns
    ):
        logger.info(
            "Renaming legacy Profit/Loss column to Scenario PnL"
        )
        results = results.rename(
            columns={
                "Profit/Loss": "Scenario PnL"
            }
        )

    scenario_columns = [
        "Scenario ID",
        "Spot Shock",
        "Volatility Shock",
        "Days Forward",
        "Rate Shock",
    ]

    scenario_details = (
        scenarios[
            [
                column
                for column in scenario_columns
                if column in scenarios.columns
            ]
        ]
        .drop_duplicates("Scenario ID")
    )

    # Remove shock columns already present before merging
    shock_columns = [
        "Spot Shock",
        "Volatility Shock",
        "Days Forward",
        "Rate Shock",
    ]

    scenario_ticker = scenario_ticker.drop(
        columns=[
            column
            for column in shock_columns
            if column in scenario_ticker.columns
        ],
        errors="ignore",
    )

    scenario_portfolio = scenario_portfolio.drop(
        columns=[
            column
            for column in shock_columns
            if column in scenario_portfolio.columns
        ],
        errors="ignore",
    )

    # Attach the market shocks to each summary table
    scenario_ticker = scenario_ticker.merge(
        scenario_details,
        on="Scenario ID",
        how="left",
    )

    scenario_portfolio = scenario_portfolio.merge(
        scenario_details,
        on="Scenario ID",
        how="left",
    )

    # Check the unchanged validation scenario
    base_result = scenario_portfolio.loc[
        scenario_portfolio["Scenario ID"].eq(
            base_scenario_id
        )
    ].copy()

    if not base_result.empty:
        base_pnl = float(
            base_result["Portfolio_PnL"].iloc[0]
        )

        base_validation_passed = bool(
            np.isclose(
                base_pnl,
                0.0,
                atol=1e-6,
            )
        )
        base_result["Base Validation Passed"] = base_validation_passed

        if base_validation_passed:
            logger.info(
                "Base scenario validation passed: portfolio P&L %.8f",
                base_pnl,
            )
        else:
            logger.warning(
                "Base scenario validation failed: portfolio P&L %.8f",
                base_pnl,
            )
    else:
        logger.warning(
            "Base scenario %s was not found in portfolio results",
            base_scenario_id,
        )

    # Find positions which could not be repriced
    invalid_scenario_rows = results.loc[
        results["Scenario Price"].isna()
        | results["Scenario PnL"].isna()
    ].copy()

    if invalid_scenario_rows.empty:
        logger.info("All scenario positions were repriced successfully")
    else:
        logger.warning(
            "%d scenario-position rows contain missing prices or P&L",
            len(invalid_scenario_rows),
        )

    # Exclude the base validation scenario from risk statistics
    scenario_distribution = (
        scenario_portfolio.loc[
            ~scenario_portfolio[
                "Scenario ID"
            ].eq(base_scenario_id)
        ]
        .copy()
    )

    if scenario_distribution.empty:
        raise ValueError(
            "No non-base scenarios are available."
        )

    # Empirical lower-tail scenario results
    pnl_5_percentile = (
        scenario_distribution["Portfolio_PnL"]
        .quantile(0.05)
    )

    pnl_1_percentile = (
        scenario_distribution["Portfolio_PnL"]
        .quantile(0.01)
    )

    expected_shortfall_5 = (
        scenario_distribution.loc[
            scenario_distribution[
                "Portfolio_PnL"
            ].le(pnl_5_percentile),
            "Portfolio_PnL",
        ]
        .mean()
    )

    # Summarise the empirical portfolio P&L distribution. These figures are
    # scenario statistics rather than formal VaR until the shock process has
    # been calibrated to a realistic time horizon and return distribution.
    scenario_risk_summary = pd.DataFrame(
        [
            {
                "Number of Scenarios": len(
                    scenario_distribution
                ),
                "Best PnL": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ].max()
                ),
                "Worst PnL": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ].min()
                ),
                "Mean PnL": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ].mean()
                ),
                "Median PnL": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ].median()
                ),
                "Probability of Loss": (
                    scenario_distribution[
                        "Portfolio_PnL"
                    ]
                    .lt(0)
                    .mean()
                ),
                "5th Percentile PnL": (
                    pnl_5_percentile
                ),
                "1st Percentile PnL": (
                    pnl_1_percentile
                ),
                "5% Expected Shortfall": (
                    expected_shortfall_5
                ),
            }
        ]
    )

    logger.info(
        "Scenario distribution analysed: worst P&L %.2f; 5th percentile %.2f; "
        "probability of loss %.2f%%",
        float(scenario_distribution["Portfolio_PnL"].min()),
        float(pnl_5_percentile),
        float(
            scenario_distribution["Portfolio_PnL"].lt(0).mean() * 100
        ),
    )

    # Find the best and worst portfolio scenarios.
    worst_scenarios = (
        scenario_distribution
        .nsmallest(
            number_to_show,
            "Portfolio_PnL",
        )
        .reset_index(drop=True)
    )

    best_scenarios = (
        scenario_distribution
        .nlargest(
            number_to_show,
            "Portfolio_PnL",
        )
        .reset_index(drop=True)
    )

    worst_scenario_id = (
        worst_scenarios["Scenario ID"]
        .iloc[0]
    )

    # Ticker contributions to the worst scenario
    worst_ticker_contributions = (
        scenario_ticker.loc[
            scenario_ticker[
                "Scenario ID"
            ].eq(worst_scenario_id)
        ]
        .sort_values("Scenario_PnL")
        .reset_index(drop=True)
    )

    # Position contributions to the worst scenario
    worst_position_contributions = (
        results.loc[
            results["Scenario ID"].eq(
                worst_scenario_id
            )
        ]
        .sort_values("Scenario PnL")
        .reset_index(drop=True)
    )

    # Find the portfolio's gross market value
    if "Gross_Market_Value" in portfolio_risk.columns:
        gross_value_column = "Gross_Market_Value"

    elif "Gross Market Value" in portfolio_risk.columns:
        gross_value_column = "Gross Market Value"

    else:
        gross_value_column = None

    portfolio_gross_value = np.nan

    if gross_value_column is not None:
        total_row = portfolio_risk.loc[
            portfolio_risk["Ticker"].eq(
                "PORTFOLIO"
            ),
            gross_value_column,
        ]

        if not total_row.empty:
            portfolio_gross_value = float(
                total_row.iloc[0]
            )
        else:
            portfolio_gross_value = float(
                portfolio_risk[
                    gross_value_column
                ].sum()
            )

    if (
        pd.notna(portfolio_gross_value)
        and portfolio_gross_value != 0
    ):
        scenario_portfolio[
            "PnL on Gross Value"
        ] = (
            scenario_portfolio[
                "Portfolio_PnL"
            ]
            / portfolio_gross_value
        )

        scenario_distribution[
            "PnL on Gross Value"
        ] = (
            scenario_distribution[
                "Portfolio_PnL"
            ]
            / portfolio_gross_value
        )
    else:
        scenario_portfolio[
            "PnL on Gross Value"
        ] = np.nan

        scenario_distribution[
            "PnL on Gross Value"
        ] = np.nan

    # Apply portfolio and ticker risk limits
    scenario_portfolio[
        "Portfolio Limit Breached"
    ] = (
        scenario_portfolio[
            "Portfolio_PnL"
        ]
        < -abs(max_portfolio_loss)
    )

    scenario_ticker[
        "Ticker Limit Breached"
    ] = (
        scenario_ticker[
            "Scenario_PnL"
        ]
        < -abs(max_ticker_loss)
    )

    # Extract scenarios which cross the configured risk limits.
    portfolio_breaches = (
        scenario_portfolio.loc[
            scenario_portfolio[
                "Portfolio Limit Breached"
            ]
        ]
        .sort_values("Portfolio_PnL")
        .reset_index(drop=True)
    )

    ticker_breaches = (
        scenario_ticker.loc[
            scenario_ticker[
                "Ticker Limit Breached"
            ]
        ]
        .sort_values("Scenario_PnL")
        .reset_index(drop=True)
    )

    if portfolio_breaches.empty:
        logger.info("No portfolio scenario breached the configured loss limit")
    else:
        logger.warning(
            "%d portfolio scenarios breached the %.2f loss limit",
            len(portfolio_breaches),
            abs(max_portfolio_loss),
        )

    if ticker_breaches.empty:
        logger.info("No ticker scenario breached the configured loss limit")
    else:
        logger.warning(
            "%d ticker-scenario rows breached the %.2f loss limit",
            len(ticker_breaches),
            abs(max_ticker_loss),
        )

    logger.info(
        "Scenario analysis complete; worst scenario is %s",
        worst_scenario_id,
    )

    return {
        "results": results,
        "scenario_ticker": scenario_ticker,
        "scenario_portfolio": scenario_portfolio,
        "scenario_distribution": scenario_distribution,
        "scenario_risk_summary": scenario_risk_summary,
        "base_result": base_result,
        "invalid_scenario_rows": invalid_scenario_rows,
        "worst_scenarios": worst_scenarios,
        "best_scenarios": best_scenarios,
        "worst_scenario_id": worst_scenario_id,
        "worst_ticker_contributions": (
            worst_ticker_contributions
        ),
        "worst_position_contributions": (
            worst_position_contributions
        ),
        "portfolio_breaches": portfolio_breaches,
        "ticker_breaches": ticker_breaches,
        "portfolio_gross_value": (
            portfolio_gross_value
        ),
    }

def calculate_greek_profit_loss(
    expanded_results: pd.DataFrame,
):
    """
    Calculate position-level Greek P&L attribution and aggregate it
    into one row per scenario.
    """

    logger.info("Starting Greek P&L attribution")

    attribution_columns = [
        "Scenario ID",
        "Contract Symbol",
        "Ticker_x",
        "Option Type",
        "Current Stock Price",
        "Shocked Spot",
        "Scenario PnL",
        "Position Delta",
        "Position Gamma",
        "Position Vega",
        "Position Theta",
        "Position Rho",
        "Spot Shock",
        "Volatility Shock",
        "Rate Shock",
        "Days Forward",
    ]

    missing_columns = [
        column
        for column in attribution_columns
        if column not in expanded_results.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing attribution columns: "
            + ", ".join(missing_columns)
        )

    attribution_results = (
        expanded_results[attribution_columns]
        .copy()
        .rename(
            columns={
                "Ticker_x": "Ticker",
            }
        )
    )

    # Position-level Greek attribution formulas

    # Dollar change in the underlying share price
    attribution_results["Spot Change"] = (
        attribution_results["Shocked Spot"]
        - attribution_results["Current Stock Price"]
    )

    # Delta P&L = position delta × change in stock price
    attribution_results["Delta PnL"] = (
        attribution_results["Position Delta"]
        * attribution_results["Spot Change"]
    )

    # Gamma P&L = 0.5 × position gamma × stock-price change squared
    attribution_results["Gamma PnL"] = (
        0.5
        * attribution_results["Position Gamma"]
        * attribution_results["Spot Change"] ** 2
    )

    # Volatility shock is stored as a decimal:
    # 0.05 represents five volatility percentage points
    attribution_results["Vega PnL"] = (
        attribution_results["Position Vega"]
        * attribution_results["Volatility Shock"]
        * 100
    )

    # Position theta is measured per calendar day
    attribution_results["Theta PnL"] = (
        attribution_results["Position Theta"]
        * attribution_results["Days Forward"]
    )

    # Rate shock is stored as a decimal:
    # 0.01 represents one interest-rate percentage point
    attribution_results["Rho PnL"] = (
        attribution_results["Position Rho"]
        * attribution_results["Rate Shock"]
        * 100
    )

    # Sum of the Greek-estimated P&L components
    attribution_results["Approximate PnL"] = (
        attribution_results["Delta PnL"]
        + attribution_results["Gamma PnL"]
        + attribution_results["Vega PnL"]
        + attribution_results["Theta PnL"]
        + attribution_results["Rho PnL"]
    )

    # Difference between full repricing and Greek approximation
    attribution_results["Residual PnL"] = (
        attribution_results["Scenario PnL"]
        - attribution_results["Approximate PnL"]
    )

    # Position-level residual percentage
    attribution_results["Residual %"] = np.where(
        attribution_results["Scenario PnL"].abs() > 1e-8,
        attribution_results["Residual PnL"].abs()
        / attribution_results["Scenario PnL"].abs(),
        np.nan,
    )


    # Scenario-level attribution
    scenario_attribution = (
        attribution_results
        .groupby(
            "Scenario ID",
            as_index=False,
        )
        .agg(
            Full_Revaluation_PnL=(
                "Scenario PnL",
                "sum",
            ),
            Delta_PnL=("Delta PnL", "sum"),
            Gamma_PnL=("Gamma PnL", "sum"),
            Vega_PnL=("Vega PnL", "sum"),
            Theta_PnL=("Theta PnL", "sum"),
            Rho_PnL=("Rho PnL", "sum"),
            Approximate_PnL=(
                "Approximate PnL",
                "sum",
            ),
            Residual_PnL=(
                "Residual PnL",
                "sum",
            ),
            Gross_Full_PnL=(
                "Scenario PnL",
                lambda values: values.abs().sum(),
            ),
            Gross_Residual=(
                "Residual PnL",
                lambda values: values.abs().sum(),
            ),
        )
    )

    # Net residual relative to the portfolio's net scenario P&L
    scenario_attribution["Net Residual %"] = np.where(
        scenario_attribution[
            "Full_Revaluation_PnL"
        ].abs() > 1e-8,
        scenario_attribution["Residual_PnL"].abs()
        / scenario_attribution[
            "Full_Revaluation_PnL"
        ].abs(),
        np.nan,
    )

    # Gross residual across all individual positions
    scenario_attribution["Gross Residual %"] = np.where(
        scenario_attribution["Gross_Full_PnL"] > 1e-8,
        scenario_attribution["Gross_Residual"]
        / scenario_attribution["Gross_Full_PnL"],
        np.nan,
    )

    # Confirm the attribution identity
    scenario_attribution["Attribution Check"] = (
        scenario_attribution["Full_Revaluation_PnL"]
        - scenario_attribution["Approximate_PnL"]
        - scenario_attribution["Residual_PnL"]
    )

    maximum_check_error = (
        scenario_attribution["Attribution Check"]
        .abs()
        .max()
    )

    if maximum_check_error <= 1e-6:
        logger.info(
            "Greek attribution accounting check passed"
        )
    else:
        logger.warning(
            "Greek attribution check failed with maximum error %.8f",
            maximum_check_error,
        )

    logger.info(
        "Greek attribution completed for %d position-scenario rows "
        "and %d scenarios",
        len(attribution_results),
        len(scenario_attribution),
    )

    return attribution_results, scenario_attribution


def analyse_greek_attribution(
    scenario_attribution: pd.DataFrame,
    attribution_results: pd.DataFrame,
    scenarios: pd.DataFrame,
    base_scenario_id: str = "BASE",
    number_to_show: int = 10,
):
    """
    Analyse the accuracy of portfolio Greek P&L attribution.

    Returns:
        scenario_attribution:
            Enriched scenario-level attribution table.

        attribution_summary:
            Overall approximation-accuracy statistics.

        small_scenario_summary:
            Accuracy statistics for relatively small shocks.

        worst_attribution_scenarios:
            Scenarios with the largest approximation errors.

        ticker_attribution:
            Greek attribution aggregated by ticker and scenario.
    """

    logger.info("Starting Greek attribution analysis")

    scenario_attribution = scenario_attribution.copy()
    attribution_results = attribution_results.copy()

    # --------------------------------------------------------------
    # 1. Attach the original scenario shocks
    # --------------------------------------------------------------

    scenario_details = (
        scenarios[
            [
                "Scenario ID",
                "Spot Shock",
                "Volatility Shock",
                "Days Forward",
                "Rate Shock",
            ]
        ]
        .drop_duplicates("Scenario ID")
    )

    # Avoid duplicated shock columns if this function is run twice
    scenario_attribution = scenario_attribution.drop(
        columns=[
            "Spot Shock",
            "Volatility Shock",
            "Days Forward",
            "Rate Shock",
        ],
        errors="ignore",
    )

    scenario_attribution = scenario_attribution.merge(
        scenario_details,
        on="Scenario ID",
        how="left",
        validate="one_to_one",
    )

    logger.info(
        "Scenario shocks attached to %d attribution rows",
        len(scenario_attribution),
    )

    # --------------------------------------------------------------
    # 2. Confirm the attribution accounting identity
    # --------------------------------------------------------------

    scenario_attribution["Attribution Check"] = (
        scenario_attribution["Full_Revaluation_PnL"]
        - scenario_attribution["Approximate_PnL"]
        - scenario_attribution["Residual_PnL"]
    )

    maximum_check_error = (
        scenario_attribution["Attribution Check"]
        .abs()
        .max()
    )

    if maximum_check_error > 1e-6:
        logger.warning(
            "Greek attribution accounting check failed: %.10f",
            maximum_check_error,
        )
    else:
        logger.info(
            "Greek attribution accounting check passed"
        )

    # --------------------------------------------------------------
    # 3. Calculate net residual percentage
    # --------------------------------------------------------------

    # This measures the residual against net portfolio P&L.
    # It can become unstable when net portfolio P&L is close to zero.
    scenario_attribution["Net Residual %"] = np.where(
        scenario_attribution[
            "Full_Revaluation_PnL"
        ].abs() > 1e-8,
        scenario_attribution["Residual_PnL"].abs()
        / scenario_attribution[
            "Full_Revaluation_PnL"
        ].abs(),
        np.nan,
    )

    # --------------------------------------------------------------
    # 4. Calculate gross residual percentage
    # --------------------------------------------------------------

    # Aggregate absolute position-level P&Ls before calculating
    # the percentage. This is more stable than the net measure.
    gross_attribution = (
        attribution_results
        .groupby(
            "Scenario ID",
            as_index=False,
        )
        .agg(
            Gross_Full_PnL=(
                "Scenario PnL",
                lambda values: values.abs().sum(),
            ),
            Gross_Residual=(
                "Residual PnL",
                lambda values: values.abs().sum(),
            ),
        )
    )

    gross_attribution["Gross Residual %"] = np.where(
        gross_attribution["Gross_Full_PnL"] > 1e-8,
        gross_attribution["Gross_Residual"]
        / gross_attribution["Gross_Full_PnL"],
        np.nan,
    )

    scenario_attribution = scenario_attribution.merge(
        gross_attribution,
        on="Scenario ID",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------------
    # 5. Validate the unchanged scenario
    # --------------------------------------------------------------

    base_attribution = scenario_attribution.loc[
        scenario_attribution["Scenario ID"].eq(
            base_scenario_id
        )
    ].copy()

    if not base_attribution.empty:
        base_columns = [
            "Full_Revaluation_PnL",
            "Delta_PnL",
            "Gamma_PnL",
            "Vega_PnL",
            "Theta_PnL",
            "Rho_PnL",
            "Approximate_PnL",
            "Residual_PnL",
        ]

        base_values = (
            base_attribution[base_columns]
            .to_numpy(dtype=float)
        )

        base_passed = np.allclose(
            base_values,
            0.0,
            atol=1e-6,
        )

        base_attribution[
            "Attribution Validation Passed"
        ] = base_passed

        if base_passed:
            logger.info(
                "BASE Greek attribution validation passed"
            )
        else:
            logger.warning(
                "BASE Greek attribution validation failed"
            )

    # --------------------------------------------------------------
    # 6. Remove BASE from accuracy statistics
    # --------------------------------------------------------------

    attribution_distribution = (
        scenario_attribution.loc[
            ~scenario_attribution[
                "Scenario ID"
            ].eq(base_scenario_id)
        ]
        .copy()
    )

    if attribution_distribution.empty:
        raise ValueError(
            "No non-base attribution scenarios are available."
        )

    # --------------------------------------------------------------
    # 7. Create overall attribution statistics
    # --------------------------------------------------------------

    attribution_summary = pd.DataFrame(
        [
            {
                "Number of Scenarios": len(
                    attribution_distribution
                ),
                "Mean Absolute Residual": (
                    attribution_distribution[
                        "Residual_PnL"
                    ]
                    .abs()
                    .mean()
                ),
                "Median Absolute Residual": (
                    attribution_distribution[
                        "Residual_PnL"
                    ]
                    .abs()
                    .median()
                ),
                "95th Percentile Absolute Residual": (
                    attribution_distribution[
                        "Residual_PnL"
                    ]
                    .abs()
                    .quantile(0.95)
                ),
                "Mean Gross Residual %": (
                    attribution_distribution[
                        "Gross Residual %"
                    ]
                    .mean()
                ),
                "Median Gross Residual %": (
                    attribution_distribution[
                        "Gross Residual %"
                    ]
                    .median()
                ),
                "95th Percentile Gross Residual %": (
                    attribution_distribution[
                        "Gross Residual %"
                    ]
                    .quantile(0.95)
                ),
            }
        ]
    )

    # --------------------------------------------------------------
    # 8. Analyse relatively small shocks separately
    # --------------------------------------------------------------

    small_scenarios = attribution_distribution.loc[
        attribution_distribution[
            "Spot Shock"
        ].abs().le(0.02)
        & attribution_distribution[
            "Volatility Shock"
        ].abs().le(0.02)
        & attribution_distribution[
            "Days Forward"
        ].le(3)
        & attribution_distribution[
            "Rate Shock"
        ].abs().le(0.0025)
    ].copy()

    small_scenario_summary = pd.DataFrame(
        [
            {
                "Number of Small Scenarios": len(
                    small_scenarios
                ),
                "Mean Absolute Residual": (
                    small_scenarios[
                        "Residual_PnL"
                    ]
                    .abs()
                    .mean()
                ),
                "Median Absolute Residual": (
                    small_scenarios[
                        "Residual_PnL"
                    ]
                    .abs()
                    .median()
                ),
                "Mean Gross Residual %": (
                    small_scenarios[
                        "Gross Residual %"
                    ]
                    .mean()
                ),
                "Median Gross Residual %": (
                    small_scenarios[
                        "Gross Residual %"
                    ]
                    .median()
                ),
            }
        ]
    )

    # --------------------------------------------------------------
    # 9. Find scenarios where Greeks performed worst
    # --------------------------------------------------------------

    worst_attribution_scenarios = (
        attribution_distribution
        .nlargest(
            number_to_show,
            "Gross Residual %",
        )
        [
            [
                "Scenario ID",
                "Spot Shock",
                "Volatility Shock",
                "Days Forward",
                "Rate Shock",
                "Full_Revaluation_PnL",
                "Delta_PnL",
                "Gamma_PnL",
                "Vega_PnL",
                "Theta_PnL",
                "Rho_PnL",
                "Approximate_PnL",
                "Residual_PnL",
                "Gross Residual %",
            ]
        ]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # 10. Create ticker-level attribution
    # --------------------------------------------------------------

    ticker_attribution = (
        attribution_results
        .groupby(
            [
                "Scenario ID",
                "Ticker",
            ],
            as_index=False,
        )
        .agg(
            Full_Revaluation_PnL=(
                "Scenario PnL",
                "sum",
            ),
            Delta_PnL=("Delta PnL", "sum"),
            Gamma_PnL=("Gamma PnL", "sum"),
            Vega_PnL=("Vega PnL", "sum"),
            Theta_PnL=("Theta PnL", "sum"),
            Rho_PnL=("Rho PnL", "sum"),
            Approximate_PnL=(
                "Approximate PnL",
                "sum",
            ),
            Residual_PnL=(
                "Residual PnL",
                "sum",
            ),
            Gross_Full_PnL=(
                "Scenario PnL",
                lambda values: values.abs().sum(),
            ),
            Gross_Residual=(
                "Residual PnL",
                lambda values: values.abs().sum(),
            ),
        )
    )

    ticker_attribution["Gross Residual %"] = np.where(
        ticker_attribution["Gross_Full_PnL"] > 1e-8,
        ticker_attribution["Gross_Residual"]
        / ticker_attribution["Gross_Full_PnL"],
        np.nan,
    )

    logger.info(
        "Greek attribution analysis completed for %d scenarios",
        len(attribution_distribution),
    )

    return {
        "scenario_attribution": scenario_attribution,
        "attribution_distribution": attribution_distribution,
        "attribution_summary": attribution_summary,
        "small_scenarios": small_scenarios,
        "small_scenario_summary": small_scenario_summary,
        "worst_attribution_scenarios": (
            worst_attribution_scenarios
        ),
        "ticker_attribution": ticker_attribution,
        "base_attribution": base_attribution,
    }


def main() -> None:
    pd.set_option("display.max_columns", None)

    VALUATION_DATE = pd.Timestamp.today().normalize()
    TARGET_DTE = 45

    tickers = [
        OptionTicker(
            symbol=symbol,
            target_dte=TARGET_DTE,
            risk_free_rate=RISK_FREE_RATE,
            forecast_lookback=FORECAST_VOL_LOOKBACK,
            buy_edge=BUY_EDGE,
            parity_tolerance=PARITY_TOLERANCE,
            valuation_date=VALUATION_DATE,
        )
        for symbol in [
            "AAPL",
            "MSFT",
            "META",
            "WMT",
            "GOOGL",
            "AMZN",
            "HCA",
        ]
    ]

    symbols = [ticker.symbol for ticker in tickers]

    logger.info(
        "Starting options pricing workflow for %d tickers",
        len(tickers),
    )

    logger.info("Downloading historical stock data")
    df = yf.download(
        symbols,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # Store each ticker's dividend yield on its own object.
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker.symbol)
            raw_dividend_yield = yf_ticker.info.get("dividendYield", 0)
            ticker.dividend_yield = float(raw_dividend_yield or 0) / 100
            ticker.refresh_derived_market_values()
        except Exception as error:
            logger.warning(
                "Failed to retrieve dividend yield for %s: %s",
                ticker.symbol,
                str(error),
            )
            ticker.dividend_yield = 0.0

    logger.info("Historical stock data downloaded: %d rows", len(df))

    df = df[["Close"]]
    returns = df["Close"].pct_change()
    returns.columns = pd.MultiIndex.from_product(
        [["Return"], returns.columns],
        names=df.columns.names,
    )
    df = pd.concat([df, returns], axis=1)
    logger.info("Daily returns calculated")

    r = df["Return"]

    rv_20 = r.rolling(20).std() * np.sqrt(252)
    rv_60 = r.rolling(60).std() * np.sqrt(252)
    rv_252 = r.rolling(252).std() * np.sqrt(252)
    logger.info("RV20, RV60 and RV252 calculated")

    abs_return = r.abs()
    squared_return = r ** 2

    ewm_vol_20 = r.ewm(span=20, adjust=False).std() * np.sqrt(252)
    ewm_vol_60 = r.ewm(span=60, adjust=False).std() * np.sqrt(252)

    mean_abs_return_5 = abs_return.rolling(5).mean()
    mean_abs_return_20 = abs_return.rolling(20).mean()
    max_abs_return_20 = abs_return.rolling(20).max()

    rv_ratio_20_60 = rv_20 / rv_60
    rv_ratio_60_252 = rv_60 / rv_252
    vol_of_vol_20 = rv_20.rolling(20).std()

    rv_20_lag1 = rv_20.shift(1)
    rv_20_lag5 = rv_20.shift(5)
    rv_60_lag1 = rv_60.shift(1)

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
            names=df.columns.names,
        )
        df = pd.concat([df, feature], axis=1)

    df = df.dropna()

    target_rv_by_symbol = {}
    for ticker in tickers:
        target_rv_by_symbol[ticker.symbol] = (
            r[ticker.symbol]
            .rolling(ticker.forecast_horizon)
            .std()
            .shift(-ticker.forecast_horizon)
            * np.sqrt(252)
        )

    target_rv = pd.DataFrame(target_rv_by_symbol)
    target_rv.columns = pd.MultiIndex.from_product(
        [["Target_RV"], target_rv.columns],
        names=df.columns.names,
    )
    df = pd.concat([df, target_rv], axis=1)
    logger.info("Target RV calculated")
    logger.info("Feature dataset created: %d usable rows", len(df))

    '''
    feature_cols = list(features.keys())
    comparison_tables = {}
    mean_holdout_tables = {}

    logger.info(
        "Starting volatility forecasting for %d ticker objects",
        len(tickers),
    )

    for ticker in tickers:
        ticker_df = df.xs(
            ticker.symbol,
            axis=1,
            level="Ticker",
        ).copy()

        mean_holdout_table = evaluate_historical_mean_holdout(
            ticker_df=ticker_df,
            ticker=ticker,
            test_fraction=0.20,
        )
        mean_holdout_tables[ticker.symbol] = mean_holdout_table
        print(f"\n{ticker.symbol} Historical Mean Holdout:")
        print(mean_holdout_table)

        comparison_table = evaluate_symbol_models(
            ticker_df=ticker_df,
            feature_cols=feature_cols,
            ticker=ticker,
            window=window,
            step=step,
            min_train_rows=min_train_rows,
        )
        comparison_tables[ticker.symbol] = comparison_table
        print(f"\n{ticker.symbol} Walk-Forward Comparison:")
        print(comparison_table)

    logger.info("Volatility forecasting complete for all symbols")
    '''

    download_option_chains(tickers)

    cleaned_options, highlighted_options, parity_results = (
        price_option_universe(
            tickers=tickers,
            data=df,
        )
    )

    print(parity_results)
    print_highlighted_options(highlighted_options)

    monte_carlo_results = run_monte_carlo_analysis(
        tickers=tickers,
        cleaned_options=cleaned_options,
        simulations=MONTE_CARLO_SIMULATIONS,
        plot_results=PLOT_MONTE_CARLO,
    )

    large_table = combine_option_tables(cleaned_options)
    print(large_table)

    findings_table = create_findings_table(large_table)
    print_final_tables(
        findings_table=findings_table,
        large_table=large_table,
    )

    # ------------------------------------------------------------------
    # Portfolio risk engine
    # ------------------------------------------------------------------
    logger.info("Starting portfolio risk-engine workflow")

    position_options = risk_engine_data_prep(large_table)
    portfolio_risk = risk_engine_summary(position_options)

    logger.info("Generating scenario shocks")
    scenario_shocks = generate_random_scenarios()

    dividend_yields = {
        ticker.symbol: ticker.dividend_yield
        for ticker in tickers
    }

    risk_free_rates = {
        ticker.symbol: ticker.risk_free_rate
        for ticker in tickers
    }

    # Map ticker-level assumptions onto every option and share position.
    position_options["Dividend Yield"] = (
        position_options["Ticker"].map(dividend_yields)
    )
    position_options["Risk Free Rate"] = (
        position_options["Ticker"].map(risk_free_rates)
    )

    logger.info("Running full-revaluation scenario engine")
    results, scenario_ticker, scenario_portfolio = run_scenario_engine(
        position_options,
        scenario_shocks,
        dividend_yields=dividend_yields
    )

    print(results)
    print(scenario_ticker)
    print(scenario_portfolio)

    logger.info("Analysing scenario and risk-limit results")
    scenario_analysis = analyse_scenario_results(
        scenarios=scenario_shocks,
        results=results,
        scenario_ticker=scenario_ticker,
        scenario_portfolio=scenario_portfolio,
        portfolio_risk=portfolio_risk,
        base_scenario_id="BASE",
        max_portfolio_loss=100_000,
        max_ticker_loss=25_000,
        number_to_show=10,
    )


    expanded_results = results.merge(position_options, left_on=["Contract Symbol"], right_on=["contractSymbol"], how="left")

    attribution_results, scenario_attribution = calculate_greek_profit_loss(expanded_results)

    attribution_analysis = analyse_greek_attribution(
        scenario_attribution=scenario_attribution,
        attribution_results=attribution_results,
        scenarios=scenario_shocks,
        base_scenario_id="BASE",
        number_to_show=10,
    )

    scenario_attribution = attribution_analysis[
        "scenario_attribution"
    ]

    attribution_summary = attribution_analysis[
        "attribution_summary"
    ]

    small_scenario_summary = attribution_analysis[
        "small_scenario_summary"
    ]

    worst_attribution_scenarios = attribution_analysis[
        "worst_attribution_scenarios"
    ]

    ticker_attribution = attribution_analysis[
        "ticker_attribution"
    ]

    print("\nGreek Attribution Summary:")
    print(attribution_summary)

    print("\nSmall-Scenario Attribution Accuracy:")
    print(small_scenario_summary)

    print("\nWorst Greek Approximation Scenarios:")
    print(worst_attribution_scenarios)

    print("\nBASE Attribution Validation:")
    print(
        attribution_analysis["base_attribution"]
    )

    logger.info("Portfolio risk-engine workflow complete")


if __name__ == "__main__":
    main()

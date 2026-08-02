
"""Volatility forecasting and leakage-aware walk-forward evaluation."""

import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from options_risk_engine.config import (
    garch_lookback,
    har_features,
    lasso_alphas,
    min_train_rows,
    pca_variance,
    ridge_alphas,
    step,
    window,
)
from options_risk_engine.domain import OptionTicker

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

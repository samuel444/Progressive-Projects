import logging

from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
    average_precision_score
)

import yfinance as yf
import numpy as np

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_validation.log", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# WALK-FORWARD VALIDATION
def walk_forward_validation(df):

    logger.info("Starting walk-forward validation")

    days = int(
        input("How many days ahead to predict? (>=1): ")
    )

    if days < 1:
        logger.error("Forecast horizon must be at least 1 day")
        raise ValueError("Days must be at least 1.")

    df = df.copy()

    df["Results"] = (
        df["Return"]
        .shift(-days)
    )

    rows_before = len(df)

    df = df.dropna()

    rows_removed = rows_before - len(df)

    if rows_removed > 0:
        logger.warning(
            "%d rows removed because of missing values",
            rows_removed
        )

    length = len(df)

    actual_values = []
    predicted_values = []

    for i in range(
        int(length * 0.25),
        length - 10,
        10
    ):

        train = df.iloc[:i]
        test = df.iloc[i:i + 10]

        # Hidden during normal INFO logging.
        # Useful if validation behaves strangely.
        logger.debug(
            "Fold: %d training rows, %d test rows",
            len(train),
            len(test)
        )

        X_train = train[["Predictor"]].values
        y_train = train["Results"].values

        X_test = test[["Predictor"]].values
        y_test = test["Results"].values

        model = LinearRegression()

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        actual_values.extend(y_test)
        predicted_values.extend(predictions)

    logger.info(
        "Walk-forward validation completed with %d predictions",
        len(predicted_values)
    )

    return actual_values, predicted_values, df

# PURGED CROSS-VALIDATION
def purged_cross_validation(df):

    logger.info("Starting purged cross-validation")

    days = int(
        input("Number of days to predict: ")
    )

    if days < 1:
        logger.error("Prediction window must be at least 1 day")
        raise ValueError("Days must be at least 1.")

    df = df.copy()

    df["Results"] = (
        (1 + df["Return"])
        .rolling(window=days)
        .apply(np.prod, raw=True)
        .shift(-days)
        - 1
    )

    rows_before = len(df)

    df = df.dropna()

    rows_removed = rows_before - len(df)

    if rows_removed > 0:
        logger.warning(
            "%d rows removed because of missing values",
            rows_removed
        )

    length = len(df)

    actual_values = []
    predicted_values = []

    for i in range(
        int(length * 0.25),
        length - 10,
        10
    ):

        train = df.iloc[:i - days]
        test = df.iloc[i:i + 10]

        logger.debug(
            "Purged fold: %d training rows, %d purged rows, %d test rows",
            len(train),
            days,
            len(test)
        )

        if len(train) == 0:
            logger.error(
                "Training set empty after purging"
            )

            raise ValueError(
                "Training set is empty."
            )

        X_train = train[["Predictor"]].values
        y_train = train["Results"].values

        X_test = test[["Predictor"]].values
        y_test = test["Results"].values

        model = LinearRegression()

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        actual_values.extend(y_test)
        predicted_values.extend(predictions)

    logger.info(
        "Purged cross-validation completed with %d predictions",
        len(predicted_values)
    )

    return actual_values, predicted_values, df

# MAIN PROGRAM
def main():

    logger.info("Program started")
    # DOWNLOAD DATA
    logger.info("Downloading AAPL data")

    df = yf.download(
        "AAPL",
        start="2021-01-01",
        end="2026-01-01",
        auto_adjust=True,
        progress=False
    )

    # CRITICAL = program cannot continue at all
    if df.empty:

        logger.critical(
            "No market data downloaded. Program cannot continue."
        )

        raise RuntimeError(
            "No market data available."
        )

    logger.info(
        "Downloaded %d rows of AAPL data",
        len(df)
    )

    df = df[
        ["Close", "Volume"]
    ]

    df["Return"] = (
        df["Close"]
        .pct_change()
    )

    # CHOOSE PREDICTOR
    print("\nChoose a predictor:")
    print("1. Previous day's return")
    print("2. Rolling mean return")
    print("3. Rolling return standard deviation")
    print("4. Momentum")
    print("5. Price-to-moving-average ratio")
    print("6. Relative volume")
    print("7. Distance from rolling high")
    print("8. Rolling return cumulative product")

    choice = int(
        input("\nEnter a number from 1 to 8: ")
    )


    if choice == 1:

        days = int(
            input("Days before today (>=1): ")
        )

        df["Predictor"] = (
            df["Return"]
            .shift(days)
        )

        predictor_name = "Previous Return"


    elif choice == 2:

        days = int(
            input("Number of rolling days: ")
        )

        df["Predictor"] = (
            df["Return"]
            .rolling(window=days)
            .mean()
        )

        predictor_name = "Rolling Mean"


    elif choice == 3:

        days = int(
            input("Number of rolling days: ")
        )

        df["Predictor"] = (
            df["Return"]
            .rolling(window=days)
            .std()
        )

        predictor_name = "Rolling Volatility"


    elif choice == 4:

        days = int(
            input("Days gap (>=1): ")
        )

        df["Predictor"] = (
            df["Close"]
            / df["Close"].shift(days)
            - 1
        )

        predictor_name = "Momentum"


    elif choice == 5:

        days = int(
            input("Number of rolling days: ")
        )

        moving_average = (
            df["Close"]
            .rolling(window=days)
            .mean()
        )

        df["Predictor"] = (
            df["Close"]
            / moving_average
            - 1
        )

        predictor_name = "Price-to-Moving-Average"


    elif choice == 6:

        days = int(
            input("Number of rolling days: ")
        )

        average_volume = (
            df["Volume"]
            .rolling(window=days)
            .mean()
        )

        df["Predictor"] = (
            df["Volume"]
            / average_volume
            - 1
        )

        predictor_name = "Relative Volume"


    elif choice == 7:

        days = int(
            input("Number of rolling days: ")
        )

        rolling_high = (
            df["Close"]
            .rolling(window=days)
            .max()
        )

        df["Predictor"] = (
            df["Close"]
            / rolling_high
            - 1
        )

        predictor_name = "Distance from Rolling High"


    elif choice == 8:

        days = int(
            input("Number of rolling days: ")
        )

        df["Predictor"] = (
            (1 + df["Return"])
            .rolling(window=days)
            .apply(np.prod, raw=True)
            - 1
        )

        predictor_name = "Rolling Cumulative Return"


    else:

        logger.error(
            "Invalid predictor choice entered: %d",
            choice
        )

        raise ValueError(
            "Choice must be a number from 1 to 8."
        )


    logger.info(
        "Predictor selected: %s, window/lag=%d",
        predictor_name,
        days
    )

    # VALIDATION METHOD
    method = input(
        "\nAre you predicting:"
        "\n1) A singular future day"
        "\n2) Rolling product over multiple days"
        "\nEnter 1 or 2: "
    )


    if method == "1":

        logger.info(
            "Validation method selected: Walk-forward"
        )

        actual_values, predicted_values, df = (
            walk_forward_validation(df)
        )


    elif method == "2":

        logger.info(
            "Validation method selected: Purged cross-validation"
        )

        actual_values, predicted_values, df = (
            purged_cross_validation(df)
        )


    else:

        logger.error(
            "Invalid validation choice entered: %s",
            method
        )

        raise ValueError(
            "Choice must be number 1 or 2."
        )


    actual_values = np.array(
        actual_values
    )

    predicted_values = np.array(
        predicted_values
    )


    if len(actual_values) == 0:

        logger.critical(
            "Validation produced no predictions. "
            "Metrics cannot be calculated."
        )

        raise RuntimeError(
            "No validation results."
        )

    # REGRESSION METRICS
    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )

    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )

    return_std = (
        df["Results"]
        .std()
    )


    if return_std == 0:

        logger.error(
            "Target standard deviation is zero. "
            "Normalized RMSE cannot be calculated."
        )

        normalized_rmse = np.nan

    else:

        normalized_rmse = (
            rmse / return_std
        )


    r_squared = r2_score(
        actual_values,
        predicted_values
    )

    # CORRELATION
    correlation = np.corrcoef(
        actual_values,
        predicted_values
    )[0, 1]


    rank_ic, _ = spearmanr(
        actual_values,
        predicted_values
    )

    # DIRECTION METRICS
    actual_direction = (
        actual_values > 0
    ).astype(int)

    predicted_direction = (
        predicted_values > 0
    ).astype(int)


    accuracy = accuracy_score(
        actual_direction,
        predicted_direction
    )


    precision = precision_score(
        actual_direction,
        predicted_direction,
        zero_division=0
    )


    positive_rate = (
        actual_direction.mean()
    )


    baseline_accuracy = max(
        positive_rate,
        1 - positive_rate
    )


    # ROC-AUC requires both positive
    # and negative observations
    if len(
        np.unique(actual_direction)
    ) < 2:

        logger.warning(
            "Only one direction exists in actual results. "
            "ROC-AUC cannot be calculated."
        )

        roc_auc = np.nan

    else:

        roc_auc = roc_auc_score(
            actual_direction,
            predicted_values
        )


    pr_auc = average_precision_score(
        actual_direction,
        predicted_values
    )

    pr_auc_baseline = (
        positive_rate
    )

    # ERROR DISTRIBUTION
    errors = (
        actual_values
        - predicted_values
    )

    mean_error = (
        errors.mean()
    )

    error_std = (
        errors.std()
    )

    median_absolute_error = np.median(
        np.abs(errors)
    )

    max_absolute_error = np.max(
        np.abs(errors)
    )

    # PREDICTION MAGNITUDE
    prediction_magnitude = np.abs(
        predicted_values
    )

    magnitude_error_correlation = np.corrcoef(
        prediction_magnitude,
        np.abs(errors)
    )[0, 1]

    # TURNOVER
    positions = np.sign(
        predicted_values
    )

    position_changes = np.sum(
        positions[1:]
        != positions[:-1]
    )


    if len(positions) <= 1:

        logger.warning(
            "Not enough predictions to calculate turnover"
        )

        turnover_rate = np.nan

    else:

        turnover_rate = (
            position_changes
            / (len(positions) - 1)
        )


    # One useful summary log rather than
    # logging every single metric separately
    logger.info(
        "Evaluation complete: "
        "RMSE=%.6f, MAE=%.6f, R2=%.4f, Accuracy=%.4f",
        rmse,
        mae,
        r_squared,
        accuracy
    )

    # RESULTS
    print("\nActual Standard Deviation:")
    print(return_std)


    if method == "1":

        print("\nWalk-Forward Validation:")


    elif method == "2":

        print("\nPurged Cross-Validation:")


    print("\nRMSE:")
    print(rmse)

    print("\nMAE:")
    print(mae)

    print("\nNormalized RMSE:")
    print(normalized_rmse)

    print("\nR-Squared:")
    print(r_squared)


    print("\nCorrelation:")
    print(correlation)

    print("\nRank IC:")
    print(rank_ic)


    print("\nDirectional Accuracy:")
    print(accuracy)

    print("\nBaseline Directional Accuracy:")
    print(baseline_accuracy)

    print("\nPrecision:")
    print(precision)


    print("\nROC-AUC:")
    print(roc_auc)

    print("\nPR-AUC:")
    print(pr_auc)

    print("\nPR-AUC Baseline:")
    print(pr_auc_baseline)


    print("\nMean Error:")
    print(mean_error)

    print("\nError Standard Deviation:")
    print(error_std)

    print("\nMedian Absolute Error:")
    print(median_absolute_error)

    print("\nLargest Absolute Error:")
    print(max_absolute_error)


    print(
        "\nPrediction Magnitude vs "
        "Absolute Error Correlation:"
    )

    print(
        magnitude_error_correlation
    )


    print("\nPosition Changes:")
    print(position_changes)

    print("\nTurnover Rate:")
    print(turnover_rate)


    logger.info(
        "Program completed successfully"
    )

# RUN PROGRAM
if __name__ == "__main__":

    try:
        main()

    except Exception:

        # Gives us the full traceback
        # for any unexpected fatal error
        logger.exception(
            "Program terminated because of an error"
        )

        raise
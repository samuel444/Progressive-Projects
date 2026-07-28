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


def walk_forward_validation(df):

    days = int(input("How many days ahead to predict? (>=1): "))

    df = df.copy()
    df["Results"] = df["Return"].shift(-days)

    df = df.dropna()
    length = len(df)

    actual_values = []
    predicted_values = []

    for i in range(int(length * 0.25), length - 10, 10):

        train = df.iloc[:i]
        test = df.iloc[i:i + 10]

        X_train = train[["Predictor"]].values
        y_train = train["Results"].values

        X_test = test[["Predictor"]].values
        y_test = test["Results"].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        actual_values.extend(y_test)
        predicted_values.extend(predictions)

    return actual_values, predicted_values, df


def purged_cross_validation(df):

    days = int(input("Number of days to predict: "))

    df = df.copy()

    df["Results"] = (
        (1 + df["Return"])
        .rolling(window=days)
        .apply(np.prod, raw=True)
        .shift(-days)
        - 1
    )

    df = df.dropna()
    length = len(df)

    actual_values = []
    predicted_values = []

    for i in range(int(length * 0.25), length - 10, 10):

        train = df.iloc[:i - days]
        test = df.iloc[i:i + 10]

        X_train = train[["Predictor"]].values
        y_train = train["Results"].values

        X_test = test[["Predictor"]].values
        y_test = test["Results"].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        actual_values.extend(y_test)
        predicted_values.extend(predictions)

    return actual_values, predicted_values, df


df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)


df = df[["Close", "Volume"]]

df["Return"] = df["Close"].pct_change()


print("\nChoose a predictor:")
print("1. Previous day's return")
print("2. Rolling mean return")
print("3. Rolling return standard deviation")
print("4. Momentum")
print("5. Price-to-moving-average ratio")
print("6. Relative volume")
print("7. Distance from rolling high")
print("8. Rolling return cumulative product")

choice = int(input("\nEnter a number from 1 to 8: "))


if choice == 1:

    days = int(input("Days before today (>=1): "))
    df["Predictor"] = df["Return"].shift(days)


elif choice == 2:

    days = int(input("Number of rolling days: "))

    df["Predictor"] = (
        df["Return"]
        .rolling(window=days)
        .mean()
    )


elif choice == 3:

    days = int(input("Number of rolling days: "))

    df["Predictor"] = (
        df["Return"]
        .rolling(window=days)
        .std()
    )


elif choice == 4:

    days = int(input("Days gap (>=1): "))

    df["Predictor"] = (
        df["Close"]
        / df["Close"].shift(days)
        - 1
    )


elif choice == 5:

    days = int(input("Number of rolling days: "))

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


elif choice == 6:

    days = int(input("Number of rolling days: "))

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


elif choice == 7:

    days = int(input("Number of rolling days: "))

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


elif choice == 8:

    days = int(input("Number of rolling days: "))

    df["Predictor"] = (
        (1 + df["Return"])
        .rolling(window=days)
        .apply(np.prod, raw=True)
        - 1
    )


else:

    raise ValueError("Choice must be a number from 1 to 8.")


method = input(
    "\nAre you predicting:"
    "\n1) A singular future day"
    "\n2) Rolling product over multiple days"
    "\nEnter 1 or 2: "
)


if method == "1":

    actual_values, predicted_values, df = walk_forward_validation(df)


elif method == "2":

    actual_values, predicted_values, df = purged_cross_validation(df)


else:

    raise ValueError("Choice must be number 1 or 2.")


actual_values = np.array(actual_values)
predicted_values = np.array(predicted_values)


# Regression metrics
rmse = np.sqrt(
    mean_squared_error(actual_values, predicted_values)
)

mae = mean_absolute_error(
    actual_values,
    predicted_values
)

return_std = df["Results"].std()

normalized_rmse = rmse / return_std

r_squared = r2_score(
    actual_values,
    predicted_values
)


# Correlation
correlation = np.corrcoef(
    actual_values,
    predicted_values
)[0, 1]


# Rank IC
rank_ic, _ = spearmanr(
    actual_values,
    predicted_values
)


# Direction metrics
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


# Baseline accuracy if we always predict
# the most common direction
positive_rate = actual_direction.mean()

baseline_accuracy = max(
    positive_rate,
    1 - positive_rate
)


# Ranking classification metrics
roc_auc = roc_auc_score(
    actual_direction,
    predicted_values
)

pr_auc = average_precision_score(
    actual_direction,
    predicted_values
)

# Baseline PR-AUC is approximately
# the proportion of positive observations
pr_auc_baseline = positive_rate



# Error distribution
errors = (
    actual_values -
    predicted_values
)

mean_error = errors.mean()

error_std = errors.std()

median_absolute_error = np.median(
    np.abs(errors)
)

max_absolute_error = np.max(
    np.abs(errors)
)


# Prediction magnitude
prediction_magnitude = np.abs(
    predicted_values
)

magnitude_error_correlation = np.corrcoef(
    prediction_magnitude,
    np.abs(errors)
)[0, 1]


# Turnover / direction changes
positions = np.sign(
    predicted_values
)

position_changes = np.sum(
    positions[1:] != positions[:-1]
)

turnover_rate = (
    position_changes /
    (len(positions) - 1)
)


# Results
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


print("\nPrediction Magnitude vs Absolute Error Correlation:")
print(magnitude_error_correlation)


print("\nPosition Changes:")
print(position_changes)

print("\nTurnover Rate:")
print(turnover_rate)
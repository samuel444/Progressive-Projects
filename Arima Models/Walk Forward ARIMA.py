import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error


df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

# Keep only closing price
df = df[["Close"]]
df["Return"] = df["Close"].pct_change()

df = df.dropna()

length = len(df)

close_price = df["Close"]

arima_predictions = []
actual_values = []


for i in range(
    int(length * 0.25),
    length - 10,
    10
):

    # Expanding training set
    train = close_price.iloc[:i]

    # Next 10 unseen observations
    test = close_price.iloc[i:i+10]


    # Fit ARIMA
    arima_model = ARIMA(
        train,
        order=(2, 1, 1)
    )

    arima_result = arima_model.fit()


    # Forecast next 10 observations
    predictions = arima_result.forecast(
        steps=len(test)
    )


    # Save predictions
    arima_predictions.extend(
        np.asarray(predictions).flatten()
    )

    # Save corresponding actual values
    actual_values.extend(
        np.asarray(test).flatten()
    )


# Convert to arrays
arima_predictions = np.array(
    arima_predictions
)

actual_values = np.array(
    actual_values
)


# Evaluate all walk-forward forecasts
arima_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        arima_predictions
    )
)



# ARMA
returns = df["Return"]

arma_predictions = []
actual_values = []

for i in range(
    int(length * 0.25),
    length - 10,
    10
):

    # Expanding training set
    train = returns.iloc[:i]

    # Next 10 unseen observations
    test = returns.iloc[i:i+10]


    # Fit ARMA
    arma_model = ARIMA(
        train,
        order=(2, 0, 1)
    )

    arma_result = arma_model.fit()


    # Forecast next 10 observations
    predictions = arma_result.forecast(
        steps=len(test)
    )


    # Save predictions
    arma_predictions.extend(
        np.asarray(predictions).flatten()
    )

    # Save corresponding actual values
    actual_values.extend(
        np.asarray(test).flatten()
    )


# Convert to arrays
arma_predictions = np.array(
    arma_predictions
)

actual_values = np.array(
    actual_values
)


# Evaluate all walk-forward forecasts
arma_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        arma_predictions
    )
)


# Exponential Smoothing
exp_predictions = []
actual_values = []

for i in range(
    int(length * 0.25),
    length - 10,
    10
):

    # Expanding training set
    train = close_price.iloc[:i]

    # Next 10 unseen observations
    test = close_price.iloc[i:i+10]


    # Fit Exp Smoothing
    exp_model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None
    )

    exp_result = exp_model.fit()


    # Forecast next 10 observations
    predictions = exp_result.forecast(
        steps=len(test)
    )


    # Save predictions
    exp_predictions.extend(
        np.asarray(predictions).flatten()
    )

    # Save corresponding actual values
    actual_values.extend(
        np.asarray(test).flatten()
    )


# Convert to arrays
exp_predictions = np.array(
    exp_predictions
)

actual_values = np.array(
    actual_values
)


# Evaluate all walk-forward forecasts
exp_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        exp_predictions
    )
)



print("\nMODEL COMPARISON")
print("-------------------------")

print("\nARIMA(2,1,1)")
print("RMSE:")
print(arima_rmse)

print("\nARMA(2,1)")
print("RMSE:")
print(arma_rmse)

print("\nExponential Smoothing")
print("RMSE:")
print(exp_rmse)
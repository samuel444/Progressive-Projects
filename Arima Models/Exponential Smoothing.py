import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error


# Download Apple data
df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

df = df[["Close"]]

df = df.dropna()

close_price = df["Close"]


# Train-test split
split = int(len(close_price) * 0.8)

train = close_price.iloc[:split]
test = close_price.iloc[split:]


# ARIMA
# Choose AR and MA orders
p = 2
q = 1

# Choose number of differences
d = 1


# ARIMA(p,d,q)
arima_model = ARIMA(
    train,
    order=(p, d, q)
)

arima_result = arima_model.fit()

print("\nARIMA Parameters:")
print(arima_result.params)


# Forecast
arima_predictions = arima_result.forecast(
    steps=len(test)
)

arima_predictions.index = test.index


# Evaluate
arima_rmse = np.sqrt(
    mean_squared_error(
        test,
        arima_predictions
    )
)

arima_nrmse = (
    arima_rmse
    / np.std(test.values)
)


print("\nARIMA RMSE:")
print(arima_rmse)

print("\nARIMA NRMSE:")
print(arima_nrmse)

print("\nARIMA BIC:")
print(arima_result.bic)


# Exponential Smoothing
exp_model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal=None
)

exp_result = exp_model.fit()


print("\nExponential Smoothing Parameters:")
print(exp_result.params)


# Forecast
exp_predictions = exp_result.forecast(
    steps=len(test)
)

exp_predictions.index = test.index


# Evaluate
exp_rmse = np.sqrt(
    mean_squared_error(
        test,
        exp_predictions
    )
)

exp_nrmse = (
    exp_rmse
    / np.std(test.values)
)


print("\nExponential Smoothing RMSE:")
print(exp_rmse)

print("\nExponential Smoothing NRMSE:")
print(exp_nrmse)


# Compare models
print("\nMODEL COMPARISON")

print(
    f"ARIMA({p},{d},{q}) "
    f"RMSE: {arima_rmse:.4f}, "
    f"NRMSE: {arima_nrmse:.4f}"
)

print(
    f"Exponential Smoothing "
    f"RMSE: {exp_rmse:.4f}, "
    f"NRMSE: {exp_nrmse:.4f}"
)


# Plot predictions
plt.plot(
    test.index,
    test,
    label="Actual Price"
)

plt.plot(
    test.index,
    arima_predictions,
    label=f"ARIMA({p},{d},{q})"
)

plt.plot(
    test.index,
    exp_predictions,
    label="Exponential Smoothing"
)

plt.xlabel("Date")
plt.ylabel("Stock Price")

plt.title(
    "ARIMA vs Exponential Smoothing"
)

plt.legend()
plt.show()
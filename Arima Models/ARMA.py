import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# 1. Download Apple data
# --------------------------------------------------

df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

df = df[["Close"]]

df["Return"] = df["Close"].pct_change()

df = df.dropna()

returns = df["Return"]


# --------------------------------------------------
# 2. Train-test split
# --------------------------------------------------

split = int(len(returns) * 0.8)

train = returns.iloc[:split]
test = returns.iloc[split:]


# Choose AR and MA orders
p = 2
q = 1


# --------------------------------------------------
# 3. AR(p)
# order = (p, 0, 0)
# --------------------------------------------------

ar_model = ARIMA(
    train,
    order=(p, 0, 0),
    trend="c"
)

ar_result = ar_model.fit()

print("\nAR Parameters:")
print(ar_result.params)


# Forecast
ar_predictions = ar_result.forecast(
    steps=len(test)
)

ar_predictions.index = test.index


# Evaluate
ar_rmse = np.sqrt(
    mean_squared_error(
        test,
        ar_predictions
    )
)

ar_nrmse = (
    ar_rmse
    / np.std(test)
)


print("\nAR RMSE:")
print(ar_rmse)

print("\nAR NRMSE:")
print(ar_nrmse)

print("\nAR BIC:")
print(ar_result.bic)


# --------------------------------------------------
# 4. MA(q)
# order = (0, 0, q)
# --------------------------------------------------

ma_model = ARIMA(
    train,
    order=(0, 0, q),
    trend="c"
)

ma_result = ma_model.fit()

print("\nMA Parameters:")
print(ma_result.params)


# Forecast
ma_predictions = ma_result.forecast(
    steps=len(test)
)

ma_predictions.index = test.index


# Evaluate
ma_rmse = np.sqrt(
    mean_squared_error(
        test,
        ma_predictions
    )
)

ma_nrmse = (
    ma_rmse
    / np.std(test)
)


print("\nMA RMSE:")
print(ma_rmse)

print("\nMA NRMSE:")
print(ma_nrmse)

print("\nMA BIC:")
print(ma_result.bic)


# --------------------------------------------------
# 5. ARMA(p, q)
# order = (p, 0, q)
# --------------------------------------------------

arma_model = ARIMA(
    train,
    order=(p, 0, q),
    trend="c"
)

arma_result = arma_model.fit()

print("\nARMA Parameters:")
print(arma_result.params)


# Forecast
arma_predictions = arma_result.forecast(
    steps=len(test)
)

arma_predictions.index = test.index


# Evaluate
arma_rmse = np.sqrt(
    mean_squared_error(
        test,
        arma_predictions
    )
)

arma_nrmse = (
    arma_rmse
    / np.std(test)
)


print("\nARMA RMSE:")
print(arma_rmse)

print("\nARMA NRMSE:")
print(arma_nrmse)

print("\nARMA BIC:")
print(arma_result.bic)


# --------------------------------------------------
# 6. Compare the three models
# --------------------------------------------------

print("\nMODEL COMPARISON")

print(
    f"AR({p})     "
    f"RMSE: {ar_rmse:.6f}, "
    f"NRMSE: {ar_nrmse:.4f}, "
    f"BIC: {ar_result.bic:.2f}"
)

print(
    f"MA({q})     "
    f"RMSE: {ma_rmse:.6f}, "
    f"NRMSE: {ma_nrmse:.4f}, "
    f"BIC: {ma_result.bic:.2f}"
)

print(
    f"ARMA({p},{q}) "
    f"RMSE: {arma_rmse:.6f}, "
    f"NRMSE: {arma_nrmse:.4f}, "
    f"BIC: {arma_result.bic:.2f}"
)


# --------------------------------------------------
# 7. Plot predictions
# --------------------------------------------------

plt.plot(
    test.index,
    test,
    label="Actual Return"
)

plt.plot(
    test.index,
    ar_predictions,
    label=f"AR({p})"
)

plt.plot(
    test.index,
    ma_predictions,
    label=f"MA({q})"
)

plt.plot(
    test.index,
    arma_predictions,
    label=f"ARMA({p},{q})"
)

plt.xlabel("Date")
plt.ylabel("Return")
plt.title("AR vs MA vs ARMA")
plt.legend()
plt.show()
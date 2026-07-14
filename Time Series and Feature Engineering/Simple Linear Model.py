import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt

df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)


# Keep only the closing price
df = df[["Close"]]

# Calculate daily returns
df["Return"] = df["Close"].pct_change()

df["Tomorrow Return"] = df["Return"].shift(-1)

df = df.dropna()
length = len(df)

train, test = df.iloc[:int(length * 0.8)], df.iloc[int(length * 0.8):]

X_train = train[["Return"]].values
y_train = train["Tomorrow Return"].values   

X_test = test[["Return"]].values
y_test = test["Tomorrow Return"].values

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

test["Predicted Return"] = model.predict(X_test)


plt.plot(X_test, y_test, "o", label="Actual Returns")
plt.plot(X_test, test["Predicted Return"], "x", label="Predicted Returns")
plt.xlabel("Today's Return")
plt.ylabel("Tomorrow's Return")
plt.legend()
plt.show()

mean_squared_error = np.mean(
    (test["Predicted Return"] - y_test) ** 2
)

rmse = np.sqrt(mean_squared_error)

baseline_predictions = np.zeros(len(y_test))

baseline_mse = np.mean(
    (baseline_predictions - y_test) ** 2
)

baseline_rmse = np.sqrt(baseline_mse)
print(f"Model RMSE: {rmse}")
print(f"Zero Baseline RMSE: {baseline_rmse}")

improvement = (
    (baseline_rmse - rmse)
    / baseline_rmse
) * 100

print(f"Improvement Over Baseline: {improvement:.2f}%")
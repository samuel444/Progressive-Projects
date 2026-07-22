import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

days = int(input("How many future days of volatility to predict: "))

df = df[["Close", "Volume"]]

df["Return"] = df["Close"].pct_change()

df["Return Lag 1"] = df["Return"].shift(1)
df["Absolute Return Lag 1"] = df["Return"].abs().shift(1)

df["Return Std 5"] = df["Return"].rolling(5).std()
df["Return Std 20"] = df["Return"].rolling(20).std()

df["Momentum 5"] = df["Close"] / df["Close"].shift(5) - 1
df["Momentum 20"] = df["Close"] / df["Close"].shift(20) - 1

df["Distance From MA 20"] = (
    df["Close"] / df["Close"].rolling(20).mean() - 1
)

df["Relative Volume 20"] = (
    df["Volume"] / df["Volume"].rolling(20).mean() - 1
)

predictor_columns = [
    "Return Lag 1",
    "Absolute Return Lag 1",
    "Return Std 5",
    "Return Std 20",
    "Momentum 5",
    "Momentum 20",
    "Distance From MA 20",
    "Relative Volume 20"
]

df["Future Volatility"] = (
    df["Return"]
    .rolling(days)
    .std()
    .shift(-days)
)

df = df.dropna()

X = df[predictor_columns]
y = df["Future Volatility"]

split = int(len(df) * 0.8)
purge = days

X_train = X.iloc[:split-purge]
X_test = X.iloc[split:]

y_train = y.iloc[:split-purge]
y_test = y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predicted_values = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
mae = mean_absolute_error(y_test, predicted_values)
r2 = r2_score(y_test, predicted_values)

print("\nForest Regressor Results:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)

print("\nFeature Importances:")

for feature, importance in zip(predictor_columns, model.feature_importances_):
    print(f"{feature:<25} {importance:.4f}")



linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\nLinear Regression Results:")
print("RMSE:", linear_rmse)
print("MAE:", linear_mae)
print("R²:", linear_r2)
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

days = int(input("Days to predict: "))
n = int(input("How many days rolling: "))

# Keep only the closing price
df = df[["Close"]]

# Calculate daily returns
df["Return"] = df["Close"].pct_change()

df["Rolling Mean"] = df["Return"].rolling(window=n).mean()

df["Return Forecast"] = (1+df["Return"]).rolling(window=days).apply(np.prod, raw=True).shift(-days) - 1

df = df.dropna()
length = len(df)


errors = []

for i in range(int(length*0.25),length-10, 10):

    train, test = df.iloc[:int(i)-days], df.iloc[int(i):int(i+10)]

    X_train = train[["Rolling Mean"]].values
    y_train = train["Return Forecast"].values   

    X_test = test[["Rolling Mean"]].values
    y_test = test["Return Forecast"].values

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)

    test["Predicted Return"] = model.predict(X_test)

    errors.extend(test["Predicted Return"] - y_test)

errors = np.array(errors)

purged_cross_validation = np.sqrt(np.mean(errors**2))

return_std = df["Return Forecast"].std()

print("\nForecast Return Standard Deviation:")
print(return_std)

print("\nPurged Cross Validation:")
print(purged_cross_validation)

print("\nNormalized RMSE:")
print(purged_cross_validation/return_std)

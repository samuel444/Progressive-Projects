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

days = int(input("Days to predict: "))

errors = []

for i in range(int(length*0.25),length-days, days):

    train, test = df.iloc[:int(i)], df.iloc[int(i):int(i+days)]

    X_train = train[["Return"]].values
    y_train = train["Tomorrow Return"].values   

    X_test = test[["Return"]].values
    y_test = test["Tomorrow Return"].values

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)

    test["Predicted Return"] = model.predict(X_test)

    errors.extend(test["Predicted Return"] - y_test)

errors = np.array(errors)
walk_forward_validation = np.sqrt(np.mean(errors**2))

print("\nWalk-Forward Validation:")
print(str(walk_forward_validation))
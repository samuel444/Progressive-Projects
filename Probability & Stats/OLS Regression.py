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

# Get x and y values
y = df["Tomorrow Return"].values

# Add a column of 1s for the intercept
X = np.column_stack((np.ones(len(df[["Return"]].values)), df[["Return"]].values))

# OLS: beta_hat = (X^T X)^(-1) X^T y
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

b0 = beta_hat[0]
b1 = beta_hat[1]

print("Manual OLS:")
print("Intercept:", b0)
print("Slope:", b1)
print("")

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(df[["Return"]].values, y)

print("Scikit results:")
print("Intercept:",model.intercept_)
print("Slope:",model.coef_[0])

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

x = df[["Return"]].values
y = df["Tomorrow Return"].values   

x = x.flatten()

x_mean = np.mean(x)
y_mean = np.mean(y)

x_deviation = x - x_mean
y_deviation = y - y_mean

covariance_term = x_deviation * y_deviation
variance_term = x_deviation ** 2

b1_numerator = np.sum(covariance_term)
b1_denominator = np.sum(variance_term)

b1 = b1_numerator / b1_denominator

b0 = y_mean - (b1 * x_mean)

print("Manual OLS:")
print("Intercept:",b0)
print("Slope:",b1)

print("")

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

print("Scikit results:")
print("Intercept:",model.intercept_)
print("Slope:",model.coef_[0])

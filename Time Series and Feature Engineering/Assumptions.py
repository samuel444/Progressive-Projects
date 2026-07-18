import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False)

# Keep only the closing price
df = df[["Close"]]

# Today's return
df["Return"] = df["Close"].pct_change()

# Return on the following trading day
df["Tomorrow Return"] = df["Return"].shift(-1)

df = df.dropna()

x = df["Return"].values
y = df["Tomorrow Return"].values

model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

predictions = model.predict(x.reshape(-1, 1))
residuals = y - predictions

df["Predicted Return"] = predictions
df["Residual"] = residuals

print()
print("Scikit-learn results:")
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])



# LINEARITY: OBSERVED DATA AND REGRESSION LINE
# Sort X so that the fitted line is drawn from left to right
sort_order = np.argsort(x)

x_sorted = x[sort_order]
predictions_sorted = predictions[sort_order]

plt.figure(figsize=(9, 5))

plt.scatter(
    x,
    y,
    alpha=0.5,
    s=18,
    label="Observed returns"
)

plt.plot(
    x_sorted,
    predictions_sorted,
    linewidth=2,
    label="OLS regression line"
)

plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)

plt.xlabel("Apple return today")
plt.ylabel("Apple return tomorrow")
plt.title("Linearity: Today's Return vs Tomorrow's Return")
plt.legend()
plt.tight_layout()
plt.show()


# CONSTANT VARIANCE:
plt.figure(figsize=(9, 5))

plt.scatter(
    predictions,
    residuals,
    alpha=0.6,
    s=18
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel("Predicted tomorrow return")
plt.ylabel("Residual")
plt.title("Residuals vs Fitted Values")
plt.tight_layout()
plt.show()


# INDEPENDENCE: RESIDUAL LAG PLOT
residuals_today = residuals[:-1]
residuals_tomorrow = residuals[1:]

residual_autocorrelation = np.corrcoef(
    residuals_today,
    residuals_tomorrow
)[0, 1]

plt.figure(figsize=(7, 6))

plt.scatter(
    residuals_today,
    residuals_tomorrow,
    alpha=0.6,
    s=18
)

plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)

plt.xlabel("Residual at time t")
plt.ylabel("Residual at time t + 1")
plt.title(
    f"Residual Lag Plot: Correlation = "
    f"{residual_autocorrelation:.3f}"
)

plt.tight_layout()
plt.show()

print()
print(
    "Lag-1 residual autocorrelation:",
    residual_autocorrelation
)


# APPROXIMATE NORMALITY: 
plt.figure(figsize=(9, 5))

plt.hist(
    residuals,
    bins=50,
    density=True,
    alpha=0.75
)

plt.axvline(
    np.mean(residuals),
    linestyle="--",
    linewidth=1,
    label="Mean residual"
)

plt.xlabel("Residual")
plt.ylabel("Density")
plt.title("Distribution of Regression Residuals")
plt.legend()
plt.tight_layout()
plt.show()



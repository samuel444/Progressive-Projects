import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


# ------------------------------------------
# 1. Download Apple stock data
# ------------------------------------------

df = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

df = df[["Close"]]

# Daily return
df["Return"] = df["Close"].pct_change()


# ------------------------------------------
# 2. Create features
# ------------------------------------------

# Lagged returns
for lag in range(1, 11):
    df[f"Lag_{lag}"] = df["Return"].shift(lag)

# Rolling average returns
df["Mean_5"] = df["Return"].rolling(5).mean()
df["Mean_10"] = df["Return"].rolling(10).mean()
df["Mean_20"] = df["Return"].rolling(20).mean()

# Rolling volatility
df["Vol_5"] = df["Return"].rolling(5).std()
df["Vol_10"] = df["Return"].rolling(10).std()
df["Vol_20"] = df["Return"].rolling(20).std()

# Target: tomorrow's return
df["Tomorrow_Return"] = df["Return"].shift(-1)

df = df.dropna()


# ------------------------------------------
# 3. Define X and y
# ------------------------------------------

features = (
    [f"Lag_{lag}" for lag in range(1, 11)]
    + [
        "Mean_5",
        "Mean_10",
        "Mean_20",
        "Vol_5",
        "Vol_10",
        "Vol_20"
    ]
)

X = df[features]
y = df["Tomorrow_Return"]

print("Number of features:", len(features))


# ------------------------------------------
# 4. Chronological train / validation / test
# ------------------------------------------

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]


# ------------------------------------------
# 5. Standard scaling
# ------------------------------------------

scaler = StandardScaler()

# Learn mean/std from TRAINING data only
X_train_scaled = scaler.fit_transform(X_train)

# Use those same train statistics
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------
# 6. OLS baseline
# ------------------------------------------

ols = LinearRegression()

ols.fit(
    X_train_scaled,
    y_train
)

ols_val_pred = ols.predict(
    X_val_scaled
)

ols_val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        ols_val_pred
    )
)


# ------------------------------------------
# 7. Ridge
# ------------------------------------------

ridge_alphas = [
    0.001,
    0.01,
    0.1,
    1,
    10
]

ridge_results = []

for alpha in ridge_alphas:

    model = Ridge(
        alpha=alpha
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    val_pred = model.predict(
        X_val_scaled
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_pred
        )
    )

    # Count coefficients that are effectively non-zero
    features_used = np.sum(
        np.abs(model.coef_) > 1e-10
    )

    ridge_results.append({
        "Alpha": alpha,
        "Validation RMSE": val_rmse,
        "Features Used": features_used,
        "Model": model
    })


# Find Ridge with lowest validation RMSE
best_ridge = min(
    ridge_results,
    key=lambda x: x["Validation RMSE"]
)


# ------------------------------------------
# 8. Lasso
# ------------------------------------------

lasso_alphas = [
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01
]

lasso_results = []

for alpha in lasso_alphas:

    model = Lasso(
        alpha=alpha,
        max_iter=100000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    val_pred = model.predict(
        X_val_scaled
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_pred
        )
    )

    features_used = np.sum(
        np.abs(model.coef_) > 1e-10
    )

    lasso_results.append({
        "Alpha": alpha,
        "Validation RMSE": val_rmse,
        "Features Used": features_used,
        "Model": model
    })


# Find Lasso with lowest validation RMSE
best_lasso = min(
    lasso_results,
    key=lambda x: x["Validation RMSE"]
)


# ------------------------------------------
# 9. Display every alpha tested
# ------------------------------------------

ridge_table = pd.DataFrame([
    {
        "Alpha": result["Alpha"],
        "Validation RMSE": result["Validation RMSE"],
        "Features Used": result["Features Used"]
    }
    for result in ridge_results
])

print("\nRIDGE RESULTS")
print(ridge_table)


lasso_table = pd.DataFrame([
    {
        "Alpha": result["Alpha"],
        "Validation RMSE": result["Validation RMSE"],
        "Features Used": result["Features Used"]
    }
    for result in lasso_results
])

print("\nLASSO RESULTS")
print(lasso_table)


# ------------------------------------------
# 10. Final comparison table
# ------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "OLS",
        "Ridge",
        "Lasso"
    ],

    "Best Alpha": [
        np.nan,
        best_ridge["Alpha"],
        best_lasso["Alpha"]
    ],

    "Validation RMSE": [
        ols_val_rmse,
        best_ridge["Validation RMSE"],
        best_lasso["Validation RMSE"]
    ],

    "Features Used": [
        np.sum(np.abs(ols.coef_) > 1e-10),
        best_ridge["Features Used"],
        best_lasso["Features Used"]
    ]
})

print("\nMODEL COMPARISON")
print(comparison)
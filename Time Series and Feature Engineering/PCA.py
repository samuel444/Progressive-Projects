import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Download data
df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

days = int(
    input("How many future days of volatility to predict: ")
)

df = df[["Close", "Volume"]]

df["Return"] = df["Close"].pct_change()

df["Return Lag 1"] = df["Return"].shift(1)
df["Absolute Return Lag 1"] = df["Return"].abs().shift(1)

df["Return Std 5"] = df["Return"].rolling(5).std()
df["Return Std 20"] = df["Return"].rolling(20).std()

df["Momentum 5"] = (
    df["Close"] / df["Close"].shift(5) - 1
)

df["Momentum 20"] = (
    df["Close"] / df["Close"].shift(20) - 1
)

df["Distance From MA 20"] = (
    df["Close"]
    / df["Close"].rolling(20).mean()
    - 1
)

df["Relative Volume 20"] = (
    df["Volume"]
    / df["Volume"].rolling(20).mean()
    - 1
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

# Target
df["Future Volatility"] = (
    df["Return"]
    .rolling(days)
    .std()
    .shift(-days)
)

df = df.dropna()


X = df[predictor_columns]
y = df["Future Volatility"]


# Chronological split with purge
split = int(len(df) * 0.8)

X_train = X.iloc[:split - days]
X_test = X.iloc[split:]

y_train = y.iloc[:split - days]
y_test = y.iloc[split:]


# Standardise
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# PCA with all components first
pca = PCA()

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)


# Explained variance
explained_variance = (
    pca.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)

variance_table = pd.DataFrame({
    "Component": [
        f"PC{i+1}"
        for i in range(len(explained_variance))
    ],

    "Explained Variance": explained_variance,

    "Cumulative Variance": cumulative_variance
})

print("\nPCA EXPLAINED VARIANCE")
print(variance_table)


# Number of PCs needed for 95%
components_95 = (
    np.argmax(cumulative_variance >= 0.95) + 1
)

print(
    "\nComponents needed for 95% variance:",
    components_95
)


# PCA again using selected number
pca_reduced = PCA(
    n_components=components_95
)

X_train_reduced = pca_reduced.fit_transform(
    X_train_scaled
)

X_test_reduced = pca_reduced.transform(
    X_test_scaled
)

# Linear regression on PCA features
pca_model = LinearRegression()

pca_model.fit(
    X_train_reduced,
    y_train
)

pca_predictions = pca_model.predict(
    X_test_reduced
)


pca_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pca_predictions
    )
)

pca_mae = mean_absolute_error(
    y_test,
    pca_predictions
)

pca_r2 = r2_score(
    y_test,
    pca_predictions
)

# Original linear regression
linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_predictions = linear_model.predict(
    X_test_scaled
)


linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# Compare
comparison = pd.DataFrame({
    "Model": [
        "Original Linear Regression",
        "PCA + Linear Regression"
    ],

    "Features": [
        len(predictor_columns),
        components_95
    ],

    "RMSE": [
        linear_rmse,
        pca_rmse
    ],

    "MAE": [
        linear_mae,
        pca_mae
    ],

    "R squared": [
        linear_r2,
        pca_r2
    ]
})

print("\nMODEL COMPARISON")
print(comparison)
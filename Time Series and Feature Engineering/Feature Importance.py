import yfinance as yf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


df = yf.download(
    "AAPL",
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

df = df[["Close", "Volume"]]
df["Return"] = df["Close"].pct_change()

df["Return Lag 1"] = df["Return"].shift(1)

df["Mean Return 5"] = df["Return"].rolling(5).mean()
df["Mean Return 20"] = df["Return"].rolling(20).mean()

df["Return Std 5"] = df["Return"].rolling(5).std()
df["Return Std 20"] = df["Return"].rolling(20).std()

df["Momentum 5"] = df["Close"] / df["Close"].shift(5) - 1
df["Momentum 20"] = df["Close"] / df["Close"].shift(20) - 1

df["Distance From MA 5"] = df["Close"] / df["Close"].rolling(5).mean() - 1
df["Distance From MA 20"] = df["Close"] / df["Close"].rolling(20).mean() - 1

df["Relative Volume 5"] = df["Volume"] / df["Volume"].rolling(5).mean() - 1
df["Relative Volume 20"] = df["Volume"] / df["Volume"].rolling(20).mean() - 1

df["Distance From High 5"] = df["Close"] / df["Close"].rolling(5).max() - 1
df["Distance From High 20"] = df["Close"] / df["Close"].rolling(20).max() - 1

predictor_columns = [
    "Return Lag 1",
    "Mean Return 5",
    "Mean Return 20",
    "Return Std 5",
    "Return Std 20",
    "Momentum 5",
    "Momentum 20",
    "Distance From MA 5",
    "Distance From MA 20",
    "Relative Volume 5",
    "Relative Volume 20",
    "Distance From High 5",
    "Distance From High 20",
    "Compounded Return 5",
    "Compounded Return 20"
]


profit = 0.05
loss = 0.05

labels = []

max_holding_period = 20

for i in range(len(df)):

    start_price = df["Close"].values[i]
    label = np.nan

    for future_price in df["Close"].values[i+1:i+max_holding_period+1]:

        change = future_price / start_price - 1

        if change >= profit:
            label = 1
            break

        if change <= -loss:
            label = 0
            break

    labels.append(label)

df["Label"] = labels

df = df.dropna()

X = df[predictor_columns]
y = df["Label"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

model = RandomForestClassifier()

model.fit(X_train, y_train)

predicted_values = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predicted_values))
print("Precision:", precision_score(y_test, predicted_values, zero_division=0))
print("Recall:", recall_score(y_test, predicted_values, zero_division=0))
print("F1 Score:", f1_score(y_test, predicted_values, zero_division=0))

print("")

for feature, importance in zip(predictor_columns,model.feature_importances_):
    print(f"{feature:<25} {importance:.4f}")

